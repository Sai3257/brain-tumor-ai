import os
import cv2
import numpy as np
from datetime import datetime

from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse, FileResponse
from django.db.models import Count
from django.db.models.functions import TruncDate
from django.utils.dateparse import parse_date
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required

from .models import MRIUpload, UserProfile
from .ml_model import predict_image
from .severity import calculate_severity
from .pdf_report import generate_pdf_report
from .dashboard_pdf import generate_dashboard_pdf


# =================================
# HOME VIEW
# =================================

def home_view(request):
    return render(request, "home.html")


# =================================
# GRAD-CAM FUNCTION
# =================================

def generate_gradcam(image_path, last_conv_layer_name="block5_conv3"):
    try:
        from .tensorflow_compat_simple import get_tensorflow_components, is_tensorflow_available
        from .ml_model import get_model

        components = get_tensorflow_components()

        if not is_tensorflow_available():
            return None

        tf = components['tf']
        np = components['np']
        model = get_model()

        if model is None:
            return None

        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_index = tf.argmax(predictions[0])
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
        heatmap = heatmap.numpy()

        heatmap = cv2.resize(heatmap, (128, 128))
        heatmap = np.uint8(255 * heatmap)  # convert float 0-1 to uint8 0-255
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        original = cv2.imread(image_path)
        original = cv2.resize(original, (128, 128))

        superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        heatmap_filename = os.path.basename(image_path).replace(".", "_heatmap.")
        heatmap_path = os.path.join(os.path.dirname(image_path), heatmap_filename)

        cv2.imwrite(heatmap_path, superimposed)

        return heatmap_filename
    except Exception as e:
        print(f"Error in generate_gradcam: {e}")
        return None


# =================================
# MAIN UPLOAD VIEW
# =================================

@login_required
def upload_mri(request):
    if request.method == "POST":
        try:
            image = request.FILES["image"]

            fs = FileSystemStorage()
            filename = fs.save(image.name, image)
            file_path = fs.path(filename)

            prediction_result = predict_image(file_path)
            tumor_type = prediction_result['tumor_type']
            confidence = prediction_result['confidence']

            severity, risk = calculate_severity(tumor_type, confidence)

            heatmap_filename = generate_gradcam(file_path)
            if heatmap_filename:
                heatmap_url = fs.url(heatmap_filename)
            else:
                heatmap_url = fs.url(filename)

            name_without_ext = os.path.splitext(filename)[0]
            pdf_filename = f"{name_without_ext}_report.pdf"
            pdf_path = os.path.join(fs.location, pdf_filename)

            generate_pdf_report(pdf_path, {
                "tumor": tumor_type,
                "confidence": round(confidence, 2),
                "severity": severity,
                "risk": risk,
                "heatmap_path": os.path.join(fs.location, heatmap_filename if heatmap_filename else filename)
            })

            pdf_url = fs.url(pdf_filename)

            MRIUpload.objects.create(
                image=filename,
                tumor_type=tumor_type,
                confidence=confidence
            )

            return render(request, "result_clean.html", {
                "tumor": tumor_type,
                "confidence": round(confidence, 2),
                "severity": severity,
                "risk": risk,
                "image": fs.url(filename),
                "heatmap": heatmap_url,
                "pdf": pdf_url
            })

        except ImportError as e:
            return render(request, "error.html", {
                "error_title": "AI Model Error",
                "error_message": f"Unable to load the AI model: {str(e)}.",
                "show_retry": True
            })
        except Exception as e:
            return render(request, "error.html", {
                "error_title": "Processing Error",
                "error_message": f"An error occurred while processing your image: {str(e)}",
                "show_retry": True
            })

    return render(request, "upload.html")


# =================================
# DASHBOARD DATA API
# =================================

@login_required
def dashboard_data(request):
    if not hasattr(request.user, 'userprofile') or request.user.userprofile.role != 'doctor':
        return JsonResponse({'error': 'Access denied'}, status=403)
    start_date = request.GET.get("start")
    end_date = request.GET.get("end")

    queryset = MRIUpload.objects.all()

    if start_date and end_date:
        queryset = queryset.filter(
            created_at__date__gte=parse_date(start_date),
            created_at__date__lte=parse_date(end_date)
        )

    tumor_data = queryset.values('tumor_type').annotate(count=Count('tumor_type'))

    trend_data = queryset.annotate(date=TruncDate('created_at')) \
        .values('date') \
        .annotate(count=Count('id')) \
        .order_by('date')

    history = queryset.order_by('-created_at')[:20]

    return JsonResponse({
        "tumor_labels": [d['tumor_type'] for d in tumor_data],
        "tumor_values": [d['count'] for d in tumor_data],
        "trend_dates": [str(t['date']) for t in trend_data],
        "trend_counts": [t['count'] for t in trend_data],
        "history": [
            {
                "tumor": h.tumor_type,
                "confidence": round(h.confidence, 2),
                "date": h.created_at.strftime("%Y-%m-%d %H:%M")
            } for h in history
        ]
    })


# =================================
# AUTH VIEWS
# =================================

def register_view(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        role = request.POST["role"]

        if User.objects.filter(username=username).exists():
            return render(request, "register.html", {"error": "Username already taken. Please choose another."})

        user = User.objects.create_user(username=username, password=password)
        profile, created = UserProfile.objects.get_or_create(user=user)
        profile.role = role
        profile.save()

        return redirect("login")

    return render(request, "register.html")


def login_view(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]

        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            profile = UserProfile.objects.get(user=user)
            if profile.role == "doctor":
                return redirect("dashboard")
            else:
                return redirect("upload")

        return render(request, "login.html", {"error": "Invalid username or password."})

    return render(request, "login.html")


def logout_view(request):
    logout(request)
    return redirect("login")


@login_required
def dashboard(request):
    if not hasattr(request.user, 'userprofile') or request.user.userprofile.role != 'doctor':
        return redirect('home')
    return render(request, "dashboard.html")


# =================================
# EXPORT DASHBOARD
# =================================

@login_required
def export_dashboard(request):
    if not hasattr(request.user, 'userprofile') or request.user.userprofile.role != 'doctor':
        return redirect('home')
    queryset = MRIUpload.objects.all()

    tumor_data = queryset.values('tumor_type').annotate(count=Count('tumor_type'))
    trend_data = queryset.annotate(date=TruncDate('created_at')) \
        .values('date') \
        .annotate(count=Count('id')) \
        .order_by('date')
    history = queryset.order_by('-created_at')[:20]

    tumor_labels = [d['tumor_type'] for d in tumor_data]
    tumor_values = [d['count'] for d in tumor_data]
    trend_dates = [str(t['date']) for t in trend_data]
    trend_counts = [t['count'] for t in trend_data]

    history_data = [
        {
            "tumor": h.tumor_type,
            "confidence": round(h.confidence, 2),
            "date": h.created_at.strftime("%Y-%m-%d %H:%M")
        } for h in history
    ]

    pdf_path = os.path.join("media", "dashboard_report.pdf")

    generate_dashboard_pdf(
        pdf_path,
        tumor_labels,
        tumor_values,
        trend_dates,
        trend_counts,
        history_data
    )

    return FileResponse(open(pdf_path, 'rb'), as_attachment=True, filename='dashboard.pdf')


# =================================
# AI DETECTION MODULE
# =================================

@login_required
def ai_detect(request):
    if request.method == "GET":
        return render(request, "ai_detect.html")

    if request.method == "POST":
        try:
            if "image" not in request.FILES:
                return JsonResponse({
                    'success': False,
                    'error': 'No image uploaded',
                    'tumor_type': None,
                    'confidence': 0,
                    'severity_level': None,
                    'risk_level': None
                })

            image = request.FILES["image"]

            fs = FileSystemStorage()
            filename = fs.save(image.name, image)
            file_path = fs.path(filename)

            output_dir = os.path.join(fs.location, "ai_results")

            try:
                import sys
                diagnosis_path = os.path.dirname(os.path.abspath(__file__))
                if diagnosis_path not in sys.path:
                    sys.path.insert(0, diagnosis_path)

                from services.ai_service import get_ai_service
                ai_service = get_ai_service()
                result = ai_service.process_image(
                    image_path=file_path,
                    save_results=True,
                    output_dir=output_dir
                )

                if result.get('success'):
                    response_data = {
                        'success': True,
                        'tumor_type': result.get('classification', {}).get('tumor_type', 'unknown'),
                        'confidence': result.get('classification', {}).get('confidence', 0),
                        'severity_level': result.get('severity', {}).get('severity_level', 'unknown'),
                        'risk_level': result.get('severity', {}).get('risk_level', 'unknown'),
                        'probabilities': result.get('classification', {}).get('probabilities', {}),
                        'original_image_url': fs.url(filename)
                    }

                    seg = result.get('segmentation', {})
                    if seg:
                        overlay_url = seg.get('overlay_url')
                        if not overlay_url and seg.get('overlay_path'):
                            overlay_filename = os.path.basename(seg['overlay_path'])
                            overlay_url = fs.url(overlay_filename)
                        response_data['segmentation'] = {
                            'overlay_url': overlay_url,
                            'overlay_available': overlay_url is not None
                        }

                    if result.get('grad_cam', {}).get('heatmap'):
                        response_data['grad_cam'] = {
                            'heatmap_available': True,
                            'heatmap_url': None
                        }
                    else:
                        response_data['grad_cam'] = {
                            'heatmap_available': False,
                            'heatmap_url': None
                        }

                    return JsonResponse(response_data)
                else:
                    return JsonResponse({
                        'success': False,
                        'error': result.get('error', 'Unknown detection error'),
                        'tumor_type': None,
                        'confidence': 0,
                        'severity_level': None,
                        'risk_level': None
                    })

            except ImportError as e:
                return JsonResponse({
                    'success': False,
                    'error': f'AI service not available: {str(e)}',
                    'tumor_type': None,
                    'confidence': 0,
                    'severity_level': None,
                    'risk_level': None
                })
            except Exception as e:
                return JsonResponse({
                    'success': False,
                    'error': f'AI processing error: {str(e)}',
                    'tumor_type': None,
                    'confidence': 0,
                    'severity_level': None,
                    'risk_level': None
                })

        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': f'AI detection error: {str(e)}',
                'tumor_type': None,
                'confidence': 0,
                'severity_level': None,
                'risk_level': None
            })


@login_required
def ai_detect_quick(request):
    if request.method == "POST":
        try:
            if "image" not in request.FILES:
                return JsonResponse({
                    'success': False,
                    'error': 'No image uploaded',
                    'tumor_type': None,
                    'confidence': 0,
                    'severity_level': None,
                    'risk_level': None
                })

            image = request.FILES["image"]

            fs = FileSystemStorage()
            filename = fs.save(image.name, image)
            file_path = fs.path(filename)

            try:
                import sys
                diagnosis_path = os.path.dirname(os.path.abspath(__file__))
                if diagnosis_path not in sys.path:
                    sys.path.insert(0, diagnosis_path)

                from services.ai_service import quick_detect_tumor
                result = quick_detect_tumor(file_path)

                if result.get('success'):
                    return JsonResponse({
                        'success': True,
                        'tumor_type': result.get('tumor_type', 'unknown'),
                        'confidence': result.get('confidence', 0),
                        'severity_level': result.get('severity_level', 'unknown'),
                        'risk_level': result.get('risk_level', 'unknown'),
                        'probabilities': result.get('probabilities', {}),
                        'original_image_url': fs.url(filename)
                    })
                else:
                    return JsonResponse({
                        'success': False,
                        'error': result.get('error', 'Unknown detection error'),
                        'tumor_type': None,
                        'confidence': 0,
                        'severity_level': None,
                        'risk_level': None
                    })

            except ImportError as e:
                return JsonResponse({
                    'success': False,
                    'error': f'AI service not available: {str(e)}',
                    'tumor_type': None,
                    'confidence': 0,
                    'severity_level': None,
                    'risk_level': None
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                return JsonResponse({
                    'success': False,
                    'error': f'Quick detection error: {str(e)}',
                    'tumor_type': None,
                    'confidence': 0,
                    'severity_level': None,
                    'risk_level': None
                })

        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': f'Server error: {str(e)}',
                'tumor_type': None,
                'confidence': 0,
                'severity_level': None,
                'risk_level': None
            })

    return JsonResponse({
        'success': False,
        'error': 'GET method not supported for this endpoint',
        'tumor_type': None,
        'confidence': 0,
        'severity_level': None,
        'risk_level': None
    })

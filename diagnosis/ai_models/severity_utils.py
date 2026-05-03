"""
Severity Prediction Module
Handles tumor severity prediction based on classification and segmentation
"""
import numpy as np
from typing import Dict, Tuple, Optional

class SeverityPredictor:
    """Enhanced tumor severity prediction with improved accuracy"""
    
    def __init__(self):
        self.severity_weights = {
            'confidence': 0.35,
            'tumor_area': 0.35,
            'tumor_type': 0.30
        }
        
        # Enhanced tumor type severity profiles
        self.tumor_type_severity = {
            'glioma': {
                'base_severity': 0.75,
                'weight': 1.15,
                'growth_rate': 'moderate',
                'treatment_complexity': 'high'
            },
            'meningioma': {
                'base_severity': 0.60,
                'weight': 1.0,
                'growth_rate': 'slow',
                'treatment_complexity': 'medium'
            },
            'notumor': {
                'base_severity': 0.0,
                'weight': 0.0,
                'growth_rate': 'none',
                'treatment_complexity': 'none'
            },
            'pituitary': {
                'base_severity': 0.68,
                'weight': 1.05,
                'growth_rate': 'variable',
                'treatment_complexity': 'medium-high'
            }
        }
        
        # Enhanced severity thresholds with more granular levels
        self.severity_thresholds = {
            'very_low': (0.0, 0.2),
            'low': (0.2, 0.4),
            'medium': (0.4, 0.6),
            'high': (0.6, 0.8),
            'very_high': (0.8, 1.0)
        }
        
        # Risk level mapping with more detail
        self.risk_mapping = {
            'very_low': 'Minimal Risk',
            'low': 'Early Stage',
            'medium': 'Moderate Risk',
            'high': 'Advanced Stage',
            'very_high': 'Critical Condition'
        }
    
    def calculate_confidence_score(self, confidence: float) -> float:
        """Calculate severity score based on confidence"""
        try:
            # Higher confidence increases severity for positive cases
            if confidence > 90:
                return 0.9
            elif confidence > 80:
                return 0.7
            elif confidence > 70:
                return 0.5
            elif confidence > 60:
                return 0.3
            else:
                return 0.1
        except Exception as e:
            print(f"ERR: Error calculating confidence score: {e}")
            return 0.5
    
    def calculate_area_score(self, tumor_percentage: float) -> float:
        """Calculate severity score based on tumor area percentage"""
        try:
            # Higher tumor percentage increases severity
            if tumor_percentage > 20:
                return 0.9
            elif tumor_percentage > 15:
                return 0.7
            elif tumor_percentage > 10:
                return 0.5
            elif tumor_percentage > 5:
                return 0.3
            else:
                return 0.1
        except Exception as e:
            print(f"ERR: Error calculating area score: {e}")
            return 0.5
    
    def calculate_type_score(self, tumor_type: str) -> float:
        """Calculate severity score based on tumor type"""
        try:
            tumor_info = self.tumor_type_severity.get(tumor_type.lower())
            if tumor_info:
                return tumor_info['base_severity'] * tumor_info['weight']
            else:
                return 0.5  # Default for unknown types
        except Exception as e:
            print(f"ERR: Error calculating type score: {e}")
            return 0.5
    
    def predict_severity(self, tumor_type: str, confidence: float, 
                        tumor_percentage: float = 0.0) -> Dict:
        """Predict overall severity"""
        try:
            # Calculate individual scores
            confidence_score = self.calculate_confidence_score(confidence)
            area_score = self.calculate_area_score(tumor_percentage)
            type_score = self.calculate_type_score(tumor_type)
            
            # Calculate weighted severity score
            severity_score = (
                confidence_score * self.severity_weights['confidence'] +
                area_score * self.severity_weights['tumor_area'] +
                type_score * self.severity_weights['tumor_type']
            )
            
            # Determine severity level
            severity_level = self._get_severity_level(severity_score)
            
            # Determine risk level
            risk_level = self._get_risk_level(severity_level, tumor_type)
            
            return {
                'severity_score': float(severity_score),
                'severity_level': severity_level,
                'risk_level': risk_level,
                'confidence_score': confidence_score,
                'area_score': area_score,
                'type_score': type_score,
                'recommendations': self._get_recommendations(severity_level, tumor_type),
                'error': None
            }
            
        except Exception as e:
            print(f"ERR: Error predicting severity: {e}")
            return {
                'severity_score': 0.0,
                'severity_level': 'unknown',
                'risk_level': 'unknown',
                'confidence_score': 0.0,
                'area_score': 0.0,
                'type_score': 0.0,
                'recommendations': [],
                'error': str(e)
            }
    
    def _get_severity_level(self, severity_score: float) -> str:
        """Get severity level based on score"""
        for level, (min_score, max_score) in self.severity_thresholds.items():
            if min_score <= severity_score <= max_score:
                return level
        return 'unknown'
    
    def _get_risk_level(self, severity_level: str, tumor_type: str) -> str:
        """Get risk level based on severity and tumor type"""
        try:
            if tumor_type.lower() == 'notumor':
                return 'Safe'
            
            risk_mapping = {
                'low': 'Early Stage',
                'medium': 'Moderate',
                'high': 'Critical'
            }
            
            return risk_mapping.get(severity_level, 'Unknown')
            
        except Exception:
            return 'Unknown'
    
    def _get_recommendations(self, severity_level: str, tumor_type: str) -> list:
        """Get medical recommendations based on severity and type"""
        try:
            if tumor_type.lower() == 'notumor':
                return [
                    "No tumor detected - continue regular checkups",
                    "Maintain healthy lifestyle",
                    "Annual MRI scans recommended"
                ]
            
            recommendations = {
                'low': [
                    "Monitor tumor growth with regular MRI scans",
                    "Consult with neurosurgeon for treatment options",
                    "Consider watchful waiting approach"
                ],
                'medium': [
                    "Immediate consultation with neuro-oncologist recommended",
                    "Consider surgical intervention",
                    "Begin treatment planning within 1-2 months"
                ],
                'high': [
                    "Urgent medical attention required",
                    "Immediate surgical consultation necessary",
                    "Begin treatment within 2-4 weeks",
                    "Consider radiation therapy and chemotherapy"
                ]
            }
            
            base_recommendations = recommendations.get(severity_level, [])
            
            # Add type-specific recommendations
            type_specific = {
                'glioma': [
                    "Consider genetic testing for personalized treatment",
                    "Discuss clinical trial options"
                ],
                'meningioma': [
                    "Hormone level monitoring may be necessary",
                    "Discuss surgical removal options"
                ],
                'pituitary': [
                    "Endocrine function testing recommended",
                    "Monitor hormone levels regularly"
                ]
            }
            
            if tumor_type.lower() in type_specific:
                base_recommendations.extend(type_specific[tumor_type.lower()])
            
            return base_recommendations
            
        except Exception as e:
            print(f"ERR: Error getting recommendations: {e}")
            return ["Consult with medical professional for personalized advice"]
    
    def get_severity_explanation(self, severity_result: Dict) -> str:
        """Get human-readable explanation of severity"""
        try:
            if severity_result['error']:
                return "Unable to determine severity due to processing error."
            
            level = severity_result['severity_level']
            score = severity_result['severity_score']
            risk = severity_result['risk_level']
            
            explanations = {
                'low': f"Low severity detected (Score: {score:.2f}). The tumor appears to be in early stages with {risk} risk level.",
                'medium': f"Medium severity detected (Score: {score:.2f}). The tumor shows moderate characteristics with {risk} risk level.",
                'high': f"High severity detected (Score: {score:.2f}). The tumor shows advanced characteristics with {risk} risk level.",
                'unknown': f"Severity assessment inconclusive (Score: {score:.2f}). Further medical evaluation recommended."
            }
            
            return explanations.get(level, "Severity assessment not available.")
            
        except Exception as e:
            print(f"ERR: Error generating severity explanation: {e}")
            return "Unable to generate severity explanation."

# Global predictor instance
_predictor = None

def get_severity_predictor():
    """Get or create severity predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = SeverityPredictor()
    return _predictor

def predict_severity(tumor_type: str, confidence: float, tumor_percentage: float = 0.0) -> Dict:
    """Convenience function for severity prediction"""
    predictor = get_severity_predictor()
    return predictor.predict_severity(tumor_type, confidence, tumor_percentage)

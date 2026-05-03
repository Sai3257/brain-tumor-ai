"""
Brain Tumor Severity Monitoring and Medical Recommendations Module
Provides comprehensive symptom-based severity assessment and medical recommendations
"""
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

class SeverityMonitor:
    """Advanced brain tumor severity monitoring with medical recommendations"""
    
    def __init__(self):
        self.severity_weights = {
            'headache_intensity': 0.15,
            'headache_frequency': 0.10,
            'seizures': 0.20,
            'vision_changes': 0.15,
            'motor_deficits': 0.15,
            'cognitive_changes': 0.10,
            'personality_changes': 0.05,
            'other_symptoms': 0.10
        }
        
        # Symptom severity levels
        self.symptom_levels = {
            'none': 0,
            'mild': 1,
            'moderate': 2,
            'severe': 3,
            'critical': 4
        }
        
        # Urgency levels
        self.urgency_levels = {
            0.0: 'Routine',
            0.3: 'Soon',
            0.6: 'Urgent',
            0.8: 'Emergency'
        }
    
    def assess_symptom_severity(self, symptoms: Dict[str, str]) -> Dict:
        """Assess overall severity based on symptoms"""
        try:
            print("Assessing symptom severity...")
            
            severity_score = 0.0
            symptom_details = {}
            critical_symptoms = []
            urgent_symptoms = []
            
            for symptom, level in symptoms.items():
                if symptom in self.severity_weights and level in self.symptom_levels:
                    weight = self.severity_weights[symptom]
                    level_value = self.symptom_levels[level]
                    contribution = weight * level_value
                    severity_score += contribution
                    
                    symptom_details[symptom] = {
                        'level': level,
                        'score': contribution,
                        'weight': weight,
                        'urgency': self._get_symptom_urgency(symptom, level)
                    }
                    
                    # Track critical and urgent symptoms
                    if level in ['severe', 'critical']:
                        if level == 'critical':
                            critical_symptoms.append(symptom)
                        else:
                            urgent_symptoms.append(symptom)
            
            # Determine overall severity level
            overall_severity = self._determine_severity_level(severity_score)
            urgency = self._determine_urgency(severity_score, critical_symptoms, urgent_symptoms)
            
            return {
                'severity_score': float(severity_score),
                'severity_level': overall_severity,
                'urgency': urgency,
                'symptom_details': symptom_details,
                'critical_symptoms': critical_symptoms,
                'urgent_symptoms': urgent_symptoms,
                'assessment_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"ERR: Error assessing symptom severity: {e}")
            return {
                'severity_score': 0.0,
                'severity_level': 'unknown',
                'urgency': 'Routine',
                'symptom_details': {},
                'critical_symptoms': [],
                'urgent_symptoms': [],
                'assessment_date': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _get_symptom_urgency(self, symptom: str, level: str) -> str:
        """Get urgency level for specific symptom"""
        urgency_map = {
            'headache': {
                'mild': 'Routine',
                'moderate': 'Soon',
                'severe': 'Urgent',
                'critical': 'Emergency'
            },
            'seizures': {
                'mild': 'Soon',
                'moderate': 'Urgent',
                'severe': 'Emergency',
                'critical': 'Emergency'
            },
            'vision_changes': {
                'mild': 'Routine',
                'moderate': 'Soon',
                'severe': 'Urgent',
                'critical': 'Emergency'
            },
            'motor_deficits': {
                'mild': 'Soon',
                'moderate': 'Urgent',
                'severe': 'Emergency',
                'critical': 'Emergency'
            },
            'cognitive_changes': {
                'mild': 'Routine',
                'moderate': 'Soon',
                'severe': 'Urgent',
                'critical': 'Emergency'
            }
        }
        
        if symptom in urgency_map and level in urgency_map[symptom]:
            return urgency_map[symptom][level]
        return 'Routine'
    
    def _determine_severity_level(self, score: float) -> str:
        """Determine overall severity level from score"""
        if score >= 3.0:
            return 'critical'
        elif score >= 2.0:
            return 'severe'
        elif score >= 1.0:
            return 'moderate'
        elif score >= 0.5:
            return 'mild'
        else:
            return 'minimal'
    
    def _determine_urgency(self, score: float, critical: List[str], urgent: List[str]) -> str:
        """Determine overall urgency"""
        if critical:
            return 'Emergency'
        elif urgent:
            return 'Urgent'
        elif score >= 2.0:
            return 'Urgent'
        elif score >= 1.0:
            return 'Soon'
        else:
            return 'Routine'
    
    def generate_medical_recommendations(self, symptoms: Dict[str, str], 
                                      severity_assessment: Dict) -> Dict:
        """Generate comprehensive medical recommendations"""
        try:
            print("Generating medical recommendations...")
            
            recommendations = []
            red_flags = []
            follow_up_schedule = []
            lifestyle_recommendations = []
            emergency_instructions = []
            
            urgency = severity_assessment.get('urgency', 'Routine')
            severity_level = severity_assessment.get('severity_level', 'minimal')
            
            # Generate recommendations based on symptoms and severity
            for symptom, level in symptoms.items():
                if level == 'critical':
                    recommendations.extend([
                        f'CRITICAL: Seek immediate emergency care for {symptom}',
                        'Call emergency services immediately',
                        'Go to nearest emergency department'
                    ])
                    red_flags.extend([
                        f'Critical {symptom} symptoms',
                        'Life-threatening condition suspected'
                    ])
                elif level == 'severe':
                    recommendations.extend([
                        f'URGENT: Seek medical attention within 24 hours for {symptom}',
                        'Contact healthcare provider immediately',
                        'Consider emergency department visit'
                    ])
                    red_flags.extend([
                        f'Severe {symptom} symptoms',
                        'Rapid progression possible'
                    ])
                elif level == 'moderate':
                    recommendations.extend([
                        f'Schedule medical appointment for {symptom} evaluation',
                        'Monitor symptoms closely',
                        'Follow up with healthcare provider'
                    ])
                elif level == 'mild':
                    recommendations.extend([
                        f'Monitor {symptom} symptoms',
                        'Keep symptom diary',
                        'Schedule routine medical follow-up'
                    ])
            
            # Add general recommendations based on severity
            if severity_level in ['severe', 'critical']:
                recommendations.extend([
                    'Arrange for caregiver support',
                    'Prepare emergency contact list',
                    'Consider hospital admission'
                ])
            
            # Generate follow-up schedule
            follow_up_schedule = self._generate_follow_up_schedule(urgency)
            
            # Generate lifestyle recommendations
            lifestyle_recommendations = self._generate_lifestyle_recommendations(severity_assessment)
            
            # Generate emergency instructions if needed
            if urgency in ['Emergency', 'Urgent']:
                emergency_instructions = self._generate_emergency_instructions(urgency)
            
            return {
                'recommendations': recommendations,
                'red_flags': red_flags,
                'follow_up_schedule': follow_up_schedule,
                'lifestyle_recommendations': lifestyle_recommendations,
                'emergency_instructions': emergency_instructions,
                'urgency': urgency,
                'generated_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"ERR: Error generating recommendations: {e}")
            return {
                'recommendations': ['Consult a healthcare provider for proper evaluation'],
                'red_flags': [],
                'follow_up_schedule': [],
                'lifestyle_recommendations': [],
                'emergency_instructions': [],
                'urgency': 'Routine',
                'generated_date': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _generate_follow_up_schedule(self, urgency: str) -> List[str]:
        """Generate follow-up schedule based on urgency"""
        schedules = {
            'Emergency': [
                'Immediate emergency department evaluation',
                'Hospital admission likely required',
                'Daily monitoring until stable'
            ],
            'Urgent': [
                'Medical evaluation within 24-48 hours',
                'Weekly follow-up until stable',
                'Regular monitoring for progression'
            ],
            'Soon': [
                'Medical appointment within 1-2 weeks',
                'Bi-weekly follow-up for first month',
                'Monthly follow-up thereafter'
            ],
            'Routine': [
                'Medical appointment within 2-4 weeks',
                'Monthly follow-up for 3 months',
                'Quarterly follow-up thereafter'
            ]
        }
        return schedules.get(urgency, schedules['Routine'])
    
    def _generate_lifestyle_recommendations(self, severity_assessment: Dict) -> List[str]:
        """Generate lifestyle recommendations based on severity"""
        severity_level = severity_assessment.get('severity_level', 'minimal')
        
        base_recommendations = [
            'Maintain regular sleep schedule (7-9 hours)',
            'Stay hydrated and eat balanced meals',
            'Avoid alcohol and smoking',
            'Manage stress through relaxation techniques',
            'Keep a symptom diary'
        ]
        
        if severity_level in ['moderate', 'severe', 'critical']:
            additional = [
                'Avoid driving until cleared by doctor',
                'Arrange for assistance with daily activities',
                'Prepare emergency contact list',
                'Consider home safety modifications',
                'Limit strenuous physical activity'
            ]
            base_recommendations.extend(additional)
        
        return base_recommendations
    
    def _generate_emergency_instructions(self, urgency: str) -> List[str]:
        """Generate emergency instructions"""
        instructions = []
        
        if urgency == 'Emergency':
            instructions.extend([
                'CALL 911 OR LOCAL EMERGENCY SERVICES IMMEDIATELY',
                'Go to nearest emergency department',
                'Do not drive yourself - call ambulance',
                'Bring all medications and medical records',
                'Inform staff about brain tumor diagnosis'
            ])
        elif urgency == 'Urgent':
            instructions.extend([
                'Seek medical attention within 24 hours',
                'Call doctor for urgent appointment',
                'Prepare to go to emergency if symptoms worsen',
                'Have someone accompany you to appointments'
            ])
        
        return instructions
    
    def create_monitoring_plan(self, symptoms: Dict[str, str], 
                             severity_assessment: Dict) -> Dict:
        """Create comprehensive monitoring plan"""
        try:
            print(" Creating monitoring plan...")
            
            urgency = severity_assessment.get('urgency', 'Routine')
            severity_level = severity_assessment.get('severity_level', 'minimal')
            
            # Determine monitoring frequency
            monitoring_frequency = self._get_monitoring_frequency(urgency, severity_level)
            
            # Create symptom tracking template
            tracking_template = self._create_tracking_template(symptoms)
            
            # Generate warning signs to watch for
            warning_signs = self._generate_warning_signs(symptoms, severity_assessment)
            
            # Create action plan for symptom changes
            action_plan = self._create_action_plan(urgency, severity_level)
            
            return {
                'monitoring_frequency': monitoring_frequency,
                'tracking_template': tracking_template,
                'warning_signs': warning_signs,
                'action_plan': action_plan,
                'created_date': datetime.now().isoformat(),
                'next_review_date': self._calculate_next_review_date(monitoring_frequency)
            }
            
        except Exception as e:
            print(f"ERR: Error creating monitoring plan: {e}")
            return {
                'monitoring_frequency': 'Weekly',
                'tracking_template': {},
                'warning_signs': [],
                'action_plan': [],
                'created_date': datetime.now().isoformat(),
                'next_review_date': (datetime.now() + timedelta(days=7)).isoformat(),
                'error': str(e)
            }
    
    def _get_monitoring_frequency(self, urgency: str, severity_level: str) -> str:
        """Determine monitoring frequency"""
        if urgency == 'Emergency':
            return 'Continuous'
        elif urgency == 'Urgent':
            return 'Daily'
        elif severity_level in ['severe', 'critical']:
            return 'Twice Daily'
        elif severity_level == 'moderate':
            return 'Weekly'
        else:
            return 'Monthly'
    
    def _create_tracking_template(self, symptoms: Dict[str, str]) -> Dict:
        """Create symptom tracking template"""
        template = {
            'date': '',
            'time': '',
            'overall_well_being': '',  # Scale 1-10
            'symptoms': {}
        }
        
        for symptom in symptoms.keys():
            template['symptoms'][symptom] = {
                'severity': '',  # none, mild, moderate, severe, critical
                'duration': '',
                'triggers': '',
                'relief_measures': '',
                'notes': ''
            }
        
        template['medications'] = {
            'taken_as_prescribed': '',
            'side_effects': '',
            'missed_doses': ''
        }
        
        template['activities'] = {
            'sleep_hours': '',
            'appetite': '',
            'exercise': '',
            'stress_level': ''
        }
        
        return template
    
    def _generate_warning_signs(self, symptoms: Dict[str, str], severity_assessment: Dict) -> List[str]:
        """Generate list of warning signs to watch for"""
        warning_signs = [
            'Sudden worsening of any symptom',
            'New symptom development',
            'Difficulty speaking or understanding',
            'Weakness or numbness on one side of body',
            'Severe headache with fever or stiff neck',
            'Confusion or personality changes',
            'Vision problems or double vision',
            'Difficulty walking or loss of balance',
            'Seizures or convulsions'
        ]
        
        return warning_signs
    
    def _create_action_plan(self, urgency: str, severity_level: str) -> List[str]:
        """Create action plan for different scenarios"""
        action_plan = []
        
        if urgency == 'Emergency':
            action_plan.extend([
                'Go to emergency department immediately',
                'Call ambulance if severe symptoms',
                'Contact neuro-oncologist on call'
            ])
        else:
            action_plan.extend([
                'Contact primary care physician',
                'Schedule appointment with neurologist',
                'Monitor symptoms closely'
            ])
        
        # Add severity-specific actions
        if severity_level in ['severe', 'critical']:
            action_plan.extend([
                'Prepare emergency contact list',
                'Arrange for caregiver support',
                'Consider hospital admission'
            ])
        
        return action_plan
    
    def _calculate_next_review_date(self, frequency: str) -> str:
        """Calculate next review date based on frequency"""
        now = datetime.now()
        
        frequency_map = {
            'Continuous': (now + timedelta(hours=1)).isoformat(),
            'Daily': (now + timedelta(days=1)).isoformat(),
            'Twice Daily': (now + timedelta(hours=12)).isoformat(),
            'Weekly': (now + timedelta(weeks=1)).isoformat(),
            'Monthly': (now + timedelta(weeks=4)).isoformat()
        }
        
        return frequency_map.get(frequency, frequency_map['Weekly'])

# Global monitor instance
_severity_monitor = None

def get_severity_monitor():
    """Get or create severity monitor instance"""
    global _severity_monitor
    if _severity_monitor is None:
        _severity_monitor = SeverityMonitor()
    return _severity_monitor

def assess_symptom_severity(symptoms: Dict[str, str]) -> Dict:
    """Assess symptom severity using monitor"""
    monitor = get_severity_monitor()
    return monitor.assess_symptom_severity(symptoms)

def generate_medical_recommendations(symptoms: Dict[str, str], severity_assessment: Dict) -> Dict:
    """Generate medical recommendations using monitor"""
    monitor = get_severity_monitor()
    return monitor.generate_medical_recommendations(symptoms, severity_assessment)

def create_monitoring_plan(symptoms: Dict[str, str], severity_assessment: Dict) -> Dict:
    """Create monitoring plan using monitor"""
    monitor = get_severity_monitor()
    return monitor.create_monitoring_plan(symptoms, severity_assessment)

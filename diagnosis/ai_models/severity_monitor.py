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
        
        # Medical recommendation database
        self.recommendations_db = self._load_medical_recommendations()
        
        # Urgency levels
        self.urgency_levels = {
            0.0: 'Routine',
            0.3: 'Soon',
            0.6: 'Urgent',
            0.8: 'Emergency'
        }
    
    def _load_medical_recommendations(self) -> Dict:
        """Load comprehensive medical recommendations database"""
        return {
            'headache': {
                'mild': {
                    'recommendations': [
                        'Monitor headache frequency and intensity',
                        'Keep a headache diary',
                        'Over-the-counter pain relievers as needed',
                        'Stay hydrated and maintain regular sleep schedule',
                        'Follow up in 2-4 weeks if symptoms persist'
                    ],
                    'urgency': 'Routine',
                    'red_flags': ['Sudden severe headache', 'Headache with fever', 'Headache after head injury']
                },
                'moderate': {
                    'recommendations': [
                        'Schedule medical appointment within 1-2 weeks',
                        'Consider prescription pain management',
                        'MRI scan recommended',
                        'Neurological consultation advised',
                        'Monitor for new symptoms'
                    ],
                    'urgency': 'Soon',
                    'red_flags': ['Progressive worsening', 'Associated with neurological symptoms']
                },
                'severe': {
                    'recommendations': [
                        'Seek immediate medical attention',
                        'Emergency department evaluation recommended',
                        'Urgent MRI/CT scan required',
                        'Neurological emergency consultation',
                        'Possible hospital admission'
                    ],
                    'urgency': 'Urgent',
                    'red_flags': ['Worst headache of life', 'Associated with confusion', 'Loss of consciousness']
                },
                'critical': {
                    'recommendations': [
                        'CALL EMERGENCY SERVICES IMMEDIATELY',
                        'Emergency department - highest priority',
                        'Immediate neuroimaging required',
                        'Neurosurgical consultation',
                        'Prepare for possible emergency intervention'
                    ],
                    'urgency': 'Emergency',
                    'red_flags': ['Sudden explosive onset', 'Associated with stroke symptoms', 'Loss of consciousness']
                }
            },
            'seizures': {
                'mild': {
                    'recommendations': [
                        'Schedule neurological consultation',
                        'EEG testing recommended',
                        'Driving restrictions may apply',
                        'Medication review with neurologist',
                        'Safety precautions at home/work'
                    ],
                    'urgency': 'Soon',
                    'red_flags': ['First seizure', 'Prolonged seizure', 'Injury during seizure']
                },
                'moderate': {
                    'recommendations': [
                        'Urgent neurological evaluation',
                        'Anti-seizure medication likely needed',
                        'Comprehensive neurological workup',
                        'Lifestyle modifications required',
                        'Regular medical follow-up essential'
                    ],
                    'urgency': 'Urgent',
                    'red_flags': ['Increasing frequency', 'Status epilepticus risk', 'New seizure types']
                },
                'severe': {
                    'recommendations': [
                        'Emergency medical evaluation required',
                        'Hospital admission likely necessary',
                        'Aggressive seizure management',
                        'Comprehensive diagnostic workup',
                        'Consider surgical evaluation'
                    ],
                    'urgency': 'Emergency',
                    'red_flags': ['Status epilepticus', 'Seizures with injury', 'New neurological deficits']
                },
                'critical': {
                    'recommendations': [
                        'IMMEDIATE EMERGENCY CARE',
                        'ICU admission likely required',
                        'Emergency anti-seizure treatment',
                        'Life-saving interventions may be needed',
                        'Prepare for critical care management'
                    ],
                    'urgency': 'Emergency',
                    'red_flags': ['Status epilepticus', 'Respiratory compromise', 'Multiple seizures']
                ]
            },
            'vision_changes': {
                'mild': {
                    'recommendations': [
                        'Comprehensive eye examination',
                        'Visual field testing',
                        'Monitor for progression',
                        'Regular follow-up with ophthalmologist',
                        'Consider neurological evaluation'
                    ],
                    'urgency': 'Routine',
                    'red_flags': ['Sudden vision loss', 'Double vision', 'Visual field defects']
                },
                'moderate': {
                    'recommendations': [
                        'Urgent ophthalmological evaluation',
                        'Neurological consultation required',
                        'MRI scan of brain and orbits',
                        'Visual field mapping',
                        'Consider tumor progression'
                    ],
                    'urgency': 'Soon',
                    'red_flags': ['Progressive vision loss', 'New visual field defects', 'Associated symptoms']
                },
                'severe': {
                    'recommendations': [
                        'Emergency medical evaluation',
                        'Urgent neuroimaging required',
                        'Ophthalmology emergency consultation',
                        'Consider increased intracranial pressure',
                        'Prepare for urgent intervention'
                    ],
                    'urgency': 'Urgent',
                    'red_flags': ['Rapid vision deterioration', 'Complete vision loss', 'Papilledema']
                },
                'critical': {
                    'recommendations': [
                        'IMMEDIATE EMERGENCY CARE',
                        'Emergency neuroimaging',
                        'Neurosurgical consultation',
                        'Treat as neurological emergency',
                        'Prepare for emergency decompression'
                    ],
                    'urgency': 'Emergency',
                    'red_flags': ['Sudden blindness', 'Severe papilledema', 'Associated with consciousness changes']
                }
            },
            'motor_deficits': {
                'mild': {
                    'recommendations': [
                        'Physical therapy evaluation',
                        'Neurological assessment',
                        'Monitor for progression',
                        'Strength and coordination exercises',
                        'Regular medical follow-up'
                    ],
                    'urgency': 'Soon',
                    'red_flags': 'New weakness, Progressive weakness, Associated with other symptoms'
                },
                'moderate': {
                    'recommendations': [
                        'Urgent neurological evaluation',
                        'Comprehensive motor assessment',
                        'MRI scan with contrast',
                        'Consider tumor progression',
                        'Rehabilitation services'
                    ],
                    'urgency': 'Urgent',
                    'red_flags': ['Rapid progression', 'Functional impairment', 'Multiple affected areas']
                },
                'severe': {
                    'recommendations': [
                        'Emergency medical evaluation',
                        'Urgent neuroimaging required',
                        'Hospital admission likely',
                        'Aggressive rehabilitation',
                        'Consider surgical intervention'
                    ],
                    'urgency': 'Emergency',
                    'red_flags': ['Severe weakness', 'Loss of function', 'Rapid deterioration']
                },
                'critical': {
                    'recommendations': [
                        'IMMEDIATE EMERGENCY CARE',
                        'Emergency neurosurgical evaluation',
                        'ICU monitoring',
                        'Prepare for emergency surgery',
                        'Life-saving interventions'
                    ],
                    'urgency': 'Emergency',
                    'red_flags': ['Paralysis', 'Respiratory muscle involvement', 'Rapid neurological decline']
                ]
            },
            'cognitive_changes': {
                'mild': {
                    'recommendations': [
                        'Neuropsychological testing',
                        'Cognitive rehabilitation',
                        'Monitor memory and concentration',
                        'Medication review',
                        'Regular cognitive assessments'
                    ],
                    'urgency': 'Routine',
                    'red_flags': ['Sudden cognitive decline', 'Memory loss affecting daily life', 'Confusion']
                },
                'moderate': {
                    'recommendations': [
                        'Comprehensive neurological evaluation',
                        'Neuropsychological assessment',
                        'Brain imaging recommended',
                        'Consider tumor effects',
                        'Cognitive therapy'
                    ],
                    'urgency': 'Soon',
                    'red_flags': ['Progressive decline', 'Affecting work/daily activities', 'Associated symptoms']
                },
                'severe': {
                    'recommendations': [
                        'Urgent neurological evaluation',
                        'Comprehensive cognitive assessment',
                        'Brain imaging with contrast',
                        'Consider increased intracranial pressure',
                        'Medical management of symptoms'
                    ],
                    'urgency': 'Urgent',
                    'red_flags': ['Severe confusion', 'Memory loss', 'Personality changes', 'Safety concerns']
                },
                'critical': {
                    'recommendations': [
                        'EMERGENCY MEDICAL CARE',
                        'Immediate neurological evaluation',
                        'Emergency brain imaging',
                        'Hospital admission required',
                        'Prepare for emergency intervention'
                    ],
                    'urgency': 'Emergency',
                    'red_flags': ['Acute confusion', 'Delirium', 'Unresponsiveness', 'Safety emergency']
                ]
            }
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
        if symptom in self.recommendations_db and level in self.recommendations_db[symptom]:
            return self.recommendations_db[symptom][level]['urgency']
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
            
            # Process each symptom
            for symptom, level in symptoms.items():
                if symptom in self.recommendations_db and level in self.recommendations_db[symptom]:
                    symptom_data = self.recommendations_db[symptom][level]
                    
                    # Add specific recommendations
                    recommendations.extend(symptom_data['recommendations'])
                    
                    # Add red flags
                    if 'red_flags' in symptom_data:
                        red_flags.extend(symptom_data['red_flags'])
            
            # Remove duplicates while preserving order
            recommendations = list(dict.fromkeys(recommendations))
            red_flags = list(dict.fromkeys(red_flags))
            
            # Generate follow-up schedule based on urgency
            urgency = severity_assessment.get('urgency', 'Routine')
            follow_up_schedule = self._generate_follow_up_schedule(urgency)
            
            # Generate lifestyle recommendations
            lifestyle_recommendations = self._generate_lifestyle_recommendations(severity_assessment)
            
            # Generate emergency instructions if needed
            if urgency in ['Emergency', 'Urgent']:
                emergency_instructions = self._generate_emergency_instructions(urgency, severity_assessment)
            
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
    
    def _generate_emergency_instructions(self, urgency: str, severity_assessment: Dict) -> List[str]:
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
        
        # Add specific red flags from current symptoms
        for symptom, level in symptoms.items():
            if symptom in self.recommendations_db and level in self.recommendations_db[symptom]:
                if 'red_flags' in self.recommendations_db[symptom][level]:
                    warning_signs.extend(self.recommendations_db[symptom][level]['red_flags'])
        
        return list(dict.fromkeys(warning_signs))  # Remove duplicates
    
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

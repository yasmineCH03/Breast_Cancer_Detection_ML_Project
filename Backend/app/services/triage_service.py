import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
from app.schemas_triage import TriageInput, TriageOutput

class TriageService:
    MAX_DAILY_CAPACITY = 8
    
    @staticmethod
    def calculate_urgency_score(probability: float, danger_flag_count: int) -> float:
        # urgency_score = (0.7 * probability) + (0.3 * (danger_flag_count / 30))
        return (0.7 * probability) + (0.3 * (danger_flag_count / 30.0))

    @staticmethod
    def get_priority_category(prediction: str, urgency_score: float) -> str:
        if prediction == "Malignant":
            return "URGENT_MALIGNANT"
        
        # Benign stratification
        if urgency_score >= 0.85:
            return "CRITIQUE"
        elif urgency_score >= 0.70:
            return "HAUT"
        elif urgency_score >= 0.50:
            return "MOYEN"
        else:
            return "FAIBLE"

    @staticmethod
    def get_max_delay_days(category: str) -> int:
        mapping = {
            "URGENT_MALIGNANT": 7,
            "CRITIQUE": 1,
            "HAUT": 7,
            "MOYEN": 30,
            "FAIBLE": 90
        }
        return mapping.get(category, 90)

    @staticmethod
    def is_weekend(date: datetime) -> bool:
        # Monday=0, Sunday=6
        return date.weekday() >= 5

    @staticmethod
    def get_next_working_day(date: datetime) -> datetime:
        next_day = date + timedelta(days=1)
        while TriageService.is_weekend(next_day):
            next_day += timedelta(days=1)
        return next_day

    @staticmethod
    def plan_patients(patients: List[TriageInput], start_date: datetime = None) -> List[TriageOutput]:
        if start_date is None:
            start_date = datetime.now()

        # 1. Enrich data with scores and categories
        enriched_patients = []
        for p in patients:
            score = TriageService.calculate_urgency_score(p.probability, p.danger_flag_count)
            category = TriageService.get_priority_category(p.prediction, score)
            max_delay = TriageService.get_max_delay_days(category)
            deadline = start_date + timedelta(days=max_delay)
            
            enriched_patients.append({
                "input": p,
                "urgency_score": score,
                "priority_category": category,
                "max_delay_days": max_delay,
                "deadline": deadline
            })

        # 2. Sort according to rules
        # Group 1: Malignant (desc score)
        malignant = [p for p in enriched_patients if p["input"].prediction == "Malignant"]
        malignant.sort(key=lambda x: x["urgency_score"], reverse=True)

        # Group 2: Benign (desc score)
        benign = [p for p in enriched_patients if p["input"].prediction == "Benign"]
        benign.sort(key=lambda x: x["urgency_score"], reverse=True)

        sorted_queue = malignant + benign

        # 3. Schedule
        schedule = []
        current_date = start_date
        # If start date is weekend, move to next working day
        if TriageService.is_weekend(current_date):
            current_date = TriageService.get_next_working_day(current_date - timedelta(days=1))
        
        appointments_today = 0
        
        # Track filled slots per day to handle large batches
        # key: date_str, value: count
        daily_load = {}

        # We will iterate through the sorted queue and find the first available slot
        # However, "next available slot" implies filling days sequentially.
        # But we must respect the "current_date" pointer to fill efficiently.
        
        # Simpler approach: Keep a pointer to the "current booking day".
        # Fill it up to 8. Then move to next working day.
        
        booking_date = current_date
        if TriageService.is_weekend(booking_date):
             booking_date = TriageService.get_next_working_day(booking_date - timedelta(days=1))

        current_day_slots_used = 0

        for p in sorted_queue:
            # Check if we need to advance the day
            if current_day_slots_used >= TriageService.MAX_DAILY_CAPACITY:
                booking_date = TriageService.get_next_working_day(booking_date)
                current_day_slots_used = 0
            
            # Assign
            if p["input"].manual_date:
                try:
                    appointment_date = datetime.strptime(p["input"].manual_date, "%Y-%m-%d")
                    # If manual date is set, we don't increment current_day_slots_used for the auto-scheduler
                    # But we should probably check if we need to advance the day for *other* patients?
                    # No, manual date is independent.
                except ValueError:
                    # Fallback if invalid format
                    appointment_date = booking_date
                    current_day_slots_used += 1
            else:
                appointment_date = booking_date
                current_day_slots_used += 1
            
            # Validate constraints
            # Check if appointment_date is within deadline
            # deadline is calculated from start_date (which is "today" or input)
            # The prompt says: "Il doit être planifié sous 7 jours calendaires max."
            # This implies the delay is relative to the time of triage/prediction?
            # Or relative to "today"?
            # Prompt: "Prediction timestamp" is an input. But usually triage happens "now".
            # I will assume the deadline is relative to the "planning run date" (start_date).
            
            days_diff = (appointment_date - start_date).days
            
            status = "CONFIRMED"
            if appointment_date > p["deadline"]:
                status = "DELAY_ALERT"

            output = TriageOutput(
                patient_id=p["input"].patient_id,
                prediction=p["input"].prediction,
                probability=p["input"].probability,
                risk_tier=p["input"].risk_tier,
                danger_flag_count=p["input"].danger_flag_count,
                urgency_score=p["urgency_score"],
                priority_category=p["priority_category"],
                appointment_date=appointment_date.strftime("%Y-%m-%d"),
                planning_status=status,
                clinical_interpretation=p["input"].clinical_interpretation,
                original_features=p["input"].original_features
            )
            schedule.append(output)
            
            if not p["input"].manual_date and current_day_slots_used >= TriageService.MAX_DAILY_CAPACITY:
                booking_date = TriageService.get_next_working_day(booking_date)
                current_day_slots_used = 0

        return schedule

    @staticmethod
    def to_csv(planning: List[TriageOutput]) -> str:
        df = pd.DataFrame([p.dict() for p in planning])
        return df.to_csv(index=False)

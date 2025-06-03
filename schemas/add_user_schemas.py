from pydantic import BaseModel, EmailStr, HttpUrl
from typing import List, Optional


class Experience(BaseModel):
    company: str  # Required
    title: str  # Required
    from_date: str  # Required, format: 'YYYY-MM'
    to: Optional[str] = None  # Optional, format: 'YYYY-MM'


class Education(BaseModel):
    education: str  # Required
    college: str  # Required
    pass_year: int  # Required


class ContactDetails(BaseModel):
    name: str  # Required
    email: EmailStr  # Required
    phone: str  # Required
    alternative_phone: Optional[str] = None
    current_city: str  # Required
    looking_for_jobs_in: List[str]  # Required
    gender: Optional[str] = None
    age: Optional[int] = None
    naukri_profile: Optional[str] = None
    linkedin_profile: Optional[str] = None
    portfolio_link: Optional[str] = None
    pan_card: str  # Required
    aadhar_card: Optional[str] = None  # Optional


class ResumeData(BaseModel):
    user_id: str
    username: str
    contact_details: ContactDetails
    total_experience: Optional[str] = None  # ✅ Already changed to string

    notice_period: Optional[str] = None  # e.g., "Immediate", "30 days"
    currency: Optional[str] = None  # e.g., "INR", "USD"
    pay_duration: Optional[str] = None  # e.g., "monthly", "yearly"
    current_salary: Optional[float] = None
    hike: Optional[float] = None
    expected_salary: Optional[float] = None  # Changed from required to optional
    skills: List[str]
    may_also_known_skills: List[str]
    labels: Optional[List[str]] = None  # Added = None for consistency
    experience: Optional[List[Experience]] = None
    academic_details: Optional[List[Education]] = None
    source: Optional[str] = None  # Source of the resume (e.g., "LinkedIn", "Naukri")
    last_working_day: Optional[str] = None  # Should be ISO format date string
    is_tier1_mba: Optional[bool] = None
    is_tier1_engineering: Optional[bool] = None
    comment: Optional[str] = None
    exit_reason: Optional[str] = None

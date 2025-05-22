# resume_api/core/vectorizer.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict


class Vectorizer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for the given text"""
        if not text or text == "N/A":
            return np.zeros(384).tolist()
        return self.model.encode(text).tolist()

    def generate_total_resume_text(self, resume_data: Dict) -> str:
        """Generate a comprehensive text representation of the entire resume"""
        sections = []

        # Add name
        if (
            "name" in resume_data
            and resume_data["name"]
            and resume_data["name"] != "N/A"
        ):
            sections.append(f"Name: {resume_data['name']}")

        # Add contact details
        if "contact_details" in resume_data:
            contact = resume_data["contact_details"]
            contact_info = []
            if contact.get("email") and contact.get("email") != "N/A":
                contact_info.append(f"Email: {contact.get('email')}")
            if contact.get("phone") and contact.get("phone") != "N/A":
                contact_info.append(f"Phone: {contact.get('phone')}")
            if contact.get("address") and contact.get("address") != "N/A":
                contact_info.append(f"Address: {contact.get('address')}")

            if contact_info:
                sections.append("Contact Details: " + "; ".join(contact_info))

        # Add education
        if "education" in resume_data and resume_data["education"]:
            edu_texts = []
            for edu in resume_data["education"]:
                parts = []
                if edu.get("degree") and edu.get("degree") != "N/A":
                    parts.append(edu.get("degree"))
                if edu.get("institution") and edu.get("institution") != "N/A":
                    parts.append(f"from {edu.get('institution')}")
                if edu.get("dates") and edu.get("dates") != "N/A":
                    parts.append(f"({edu.get('dates')})")

                if parts:
                    edu_texts.append(" ".join(parts))

            if edu_texts:
                sections.append("Education: " + ". ".join(edu_texts))

        # Add experience
        if "experience" in resume_data and resume_data["experience"]:
            exp_texts = []
            for exp in resume_data["experience"]:
                parts = []
                if exp.get("title") and exp.get("title") != "N/A":
                    parts.append(exp.get("title"))
                if exp.get("company") and exp.get("company") != "N/A":
                    parts.append(f"at {exp.get('company')}")
                if exp.get("duration") and exp.get("duration") != "N/A":
                    parts.append(f"for {exp.get('duration')}")

                if parts:
                    exp_texts.append(" ".join(parts))

            if exp_texts:
                sections.append("Experience: " + ". ".join(exp_texts))

        # Add total experience
        if (
            "total_experience" in resume_data
            and resume_data["total_experience"]
            and resume_data["total_experience"] != "N/A"
        ):
            sections.append(f"Total Experience: {resume_data['total_experience']}")

        # Add skills
        if "skills" in resume_data and resume_data["skills"]:
            skills_text = (
                ", ".join(resume_data["skills"])
                if isinstance(resume_data["skills"], list)
                else resume_data["skills"]
            )
            if skills_text and skills_text != "N/A":
                sections.append(f"Skills: {skills_text}")

        # Add projects
        if "projects" in resume_data and resume_data["projects"]:
            proj_texts = []
            for proj in resume_data["projects"]:
                project_parts = []
                if proj.get("name") and proj.get("name") != "N/A":
                    project_parts.append(proj.get("name"))

                if proj.get("description") and proj.get("description") != "N/A":
                    project_parts.append(proj.get("description"))

                tech_text = ""
                if proj.get("technologies"):
                    tech_text = (
                        ", ".join(proj.get("technologies"))
                        if isinstance(proj.get("technologies"), list)
                        else proj.get("technologies")
                    )
                    if tech_text and tech_text != "N/A":
                        project_parts.append(f"Technologies: {tech_text}")

                if proj.get("role") and proj.get("role") != "N/A":
                    project_parts.append(f"Role: {proj.get('role')}")

                if project_parts:
                    proj_texts.append(". ".join(project_parts))

            if proj_texts:
                sections.append("Projects: " + ". ".join(proj_texts))

        # Add certifications
        if "certifications" in resume_data and resume_data["certifications"]:
            cert_text = (
                ", ".join(resume_data["certifications"])
                if isinstance(resume_data["certifications"], list)
                else resume_data["certifications"]
            )
            if cert_text and cert_text != "N/A":
                sections.append(f"Certifications: {cert_text}")

        # Join all sections with line breaks
        return "\n\n".join(sections)

    def generate_resume_embeddings(self, resume_data: Dict) -> Dict:
        """Generate embeddings for searchable fields in resume and a combined vector for the entire resume"""
        resume_with_vectors = resume_data.copy()

        # Generate vector for skills (join list to string)
        skills_text = ""
        if "skills" in resume_data and resume_data["skills"]:
            skills_text = (
                ", ".join(resume_data["skills"])
                if isinstance(resume_data["skills"], list)
                else resume_data["skills"]
            )
            resume_with_vectors["skills_vector"] = self.generate_embedding(skills_text)

        # Generate vector for experience text
        experience_text = ""
        if "experience" in resume_data and resume_data["experience"]:
            experience_texts = []
            for exp in resume_data["experience"]:
                exp_text = f"{exp.get('title', '')} at {exp.get('company', '')} for {exp.get('duration', '')}"
                experience_texts.append(exp_text)
            experience_text = ". ".join(experience_texts)
            resume_with_vectors["experience_text_vector"] = self.generate_embedding(
                experience_text
            )

        # Generate vector for education text
        education_text = ""
        if "education" in resume_data and resume_data["education"]:
            education_texts = []
            for edu in resume_data["education"]:
                edu_text = f"{edu.get('degree', '')} from {edu.get('institution', '')} ({edu.get('dates', '')})"
                education_texts.append(edu_text)
            education_text = ". ".join(education_texts)
            resume_with_vectors["education_text_vector"] = self.generate_embedding(
                education_text
            )

        # Generate vector for projects text
        projects_text = ""
        if "projects" in resume_data and resume_data["projects"]:
            project_texts = []
            for proj in resume_data["projects"]:
                tech_text = (
                    ", ".join(proj.get("technologies", []))
                    if isinstance(proj.get("technologies"), list)
                    else proj.get("technologies", "")
                )
                proj_text = f"{proj.get('name', '')}: {proj.get('description', '')}. Technologies: {tech_text}. Role: {proj.get('role', '')}"
                project_texts.append(proj_text)
            projects_text = ". ".join(project_texts)
            resume_with_vectors["projects_text_vector"] = self.generate_embedding(
                projects_text
            )

        # Generate vector for the entire resume

        # NEW: Generate total resume text and its vector
        total_resume_text = self.generate_total_resume_text(resume_data)
        resume_with_vectors["total_resume_text"] = total_resume_text
        resume_with_vectors["total_resume_vector"] = self.generate_embedding(
            total_resume_text
        )

        return resume_with_vectors

    def generate_total_res_textv2(self, resume_dict: Dict) -> str:
        combined_resume = f"""
                        RESUME

                        PERSONAL INFORMATION
                        -------------------
                        Name: {resume_dict['name']}
                        Email: {resume_dict['email']}
                        Phone: {resume_dict['phone_number']}
                        Alternative Phone: {resume_dict.get('alternative_number', 'N/A')}
                        Current City: {resume_dict['current_city']}
                        Looking for jobs in: {resume_dict['looking_for_jobs_in']}
                        Gender: {resume_dict['gender']}
                        Age: {resume_dict['age']}

                        PROFESSIONAL SUMMARY
                        -------------------
                        Total Experience: {resume_dict['total_experience']} years
                        Notice Period: {resume_dict['notice_period']} days
                        Current Salary: {resume_dict['currency']} {resume_dict['current_salary']} ({resume_dict['pay_duration']})
                        Expected Salary: {resume_dict['currency']} {resume_dict['expected_salary']} ({resume_dict['pay_duration']})
                        Hike Expected: {resume_dict['hike']}%
                        Last Working Day: {resume_dict.get('last_working_day', 'N/A')}
                        Exit Reason: {resume_dict.get('exit_reason', 'N/A')}

                        SKILLS
                        ------
                        Primary Skills: {', '.join(resume_dict['skills'])}
                        Additional Skills: {', '.join(resume_dict['may_also_known_skills']) if resume_dict['may_also_known_skills'] else 'N/A'}
                        Labels: {', '.join(resume_dict['labels']) if resume_dict['labels'] else 'N/A'}

                        PROFESSIONAL EXPERIENCE
                        ----------------------
                        {chr(10).join([f'''
                        Company: {exp['company']}
                        Title: {exp['title']}
                        Duration: {exp['from_date']} to {exp['until'] if exp['until'] else 'Present'}
                        ''' for exp in resume_dict['experience']])}

                        EDUCATION
                        ---------
                        {chr(10).join([f'''
                        Degree: {edu['education']}
                        College: {edu['college']}
                        Pass Year: {edu['pass_year']}
                        ''' for edu in resume_dict['academic_details']])}

                        ADDITIONAL INFORMATION
                        ---------------------
                        Tier 1 MBA: {'Yes' if resume_dict['is_tier1_mba'] else 'No'}
                        Tier 1 Engineering: {'Yes' if resume_dict['is_tier1_engineering'] else 'No'}
                        Comments: {resume_dict.get('comment', 'N/A')}

                        PROFESSIONAL LINKS
                        -----------------
                        Naukri Profile: {resume_dict.get('naukri_profile', 'N/A')}
                        LinkedIn Profile: {resume_dict.get('linkedin_profile', 'N/A')}
                        Portfolio: {resume_dict.get('portfolio_link', 'N/A')}

                        ORIGINAL RESUME TEXT
                        -------------------
                {resume_dict.get('total_resume_text', 'N/A')}
                """

        print(combined_resume)
        return combined_resume

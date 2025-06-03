# resume_api/database/operations.py
from pymongo.collection import Collection
from bson import ObjectId
from typing import Dict, Any
from embeddings.vectorizer import Vectorizer
from core.helpers import format_resume
from fastapi import HTTPException
from typing import List


class ResumeOperations:
    def __init__(self, collection: Collection, vectorizer: Vectorizer):
        self.collection = collection
        self.vectorizer = vectorizer

    def create_resume(self, resume_data: Dict[str, Any]) -> Dict[str, str]:
        """Create a new resume with vector embeddings"""
        resume_with_vectors = self.vectorizer.generate_resume_embeddings(resume_data)
        result = self.collection.insert_one(resume_with_vectors)
        return {
            "id": str(result.inserted_id),
            "message": "Resume created successfully with vector embeddings",
        }

    def update_resume(
        self, resume_id: str, resume_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """Update a resume by ID with vector embeddings"""
        existing = self.collection.find_one({"_id": ObjectId(resume_id)})
        if not existing:
            raise HTTPException(status_code=404, detail="Resume not found")

        resume_with_vectors = self.vectorizer.generate_resume_embeddings(resume_data)
        result = self.collection.update_one(
            {"_id": ObjectId(resume_id)}, {"$set": resume_with_vectors}
        )

        if result.modified_count == 1:
            return {"message": "Resume updated successfully with vector embeddings"}
        return {"message": "No changes made to the resume"}

    def get_resume(self, resume_id: str) -> Dict:
        """Get a resume by ID"""
        resume = self.collection.find_one({"_id": ObjectId(resume_id)})
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        return format_resume(resume)

    def delete_resume(self, resume_id: str) -> Dict:
        """Delete a resume by ID"""
        result = self.collection.delete_one({"_id": ObjectId(resume_id)})
        if result.deleted_count != 1:
            raise HTTPException(status_code=404, detail="Resume not found")
        return {"message": "Resume deleted successfully"}

    def list_resumes(self, skip: int = 0, limit: int = 10) -> List[Dict]:
        """List all resumes with pagination"""
        cursor = self.collection.find().skip(skip).limit(limit)
        return [format_resume(doc) for doc in cursor]

    def update_all_vector_embeddings(self) -> Dict:
        """Update vector embeddings for all resumes"""
        resumes = list(self.collection.find({}))
        updated_count = 0

        for resume in resumes:
            resume_with_vectors = self.vectorizer.generate_resume_embeddings(resume)
            result = self.collection.update_one(
                {"_id": resume["_id"]},
                {
                    "$set": {
                        "skills_vector": resume_with_vectors.get("skills_vector"),
                        "experience_text_vector": resume_with_vectors.get(
                            "experience_text_vector"
                        ),
                        "academic_details_vector": resume_with_vectors.get(
                            "academic_details_vector"
                        ),
                        "combined_resume_vector": resume_with_vectors.get(
                            "combined_resume_vector"
                        ),
                    }
                },
            )
            if result.modified_count > 0:
                updated_count += 1

        return {"message": f"Updated vector embeddings for {updated_count} resumes"}


# ...existing code...


class SkillsTitlesOperations:
    def __init__(self, collection: Collection):
        self.collection = collection

    def _preprocess_text(self, text: str) -> str:
        """Preprocess the text by removing extra spaces and converting to lowercase"""
        return text.strip().lower()

    def add_skill(self, skill: str) -> Dict[str, str]:
        """Add a new skill if it doesn't exist"""
        processed_skill = self._preprocess_text(skill)

        # Check if skill already exists
        existing_skill = self.collection.find_one(
            {"type": "skill", "value": processed_skill}
        )
        if existing_skill:
            return {"message": f"Skill '{skill}' already exists"}

        result = self.collection.insert_one(
            {
                "type": "skill",
                "value": processed_skill,
            }
        )

        return {"message": f"Skill '{skill}' added successfully"}

    def add_title(self, title: str) -> Dict[str, str]:
        """Add a new title if it doesn't exist"""
        processed_title = self._preprocess_text(title)

        # Check if title already exists
        existing_title = self.collection.find_one(
            {"type": "title", "value": processed_title}
        )
        if existing_title:
            return {"message": f"Title '{title}' already exists"}

        result = self.collection.insert_one(
            {
                "type": "title",
                "value": processed_title,
            }
        )

        return {"message": f"Title '{title}' added successfully"}

    def add_multiple_skills(self, skills: List[str]) -> Dict[str, Any]:
        """Add multiple skills at once"""
        added_count = 0
        existing_count = 0

        for skill in skills:
            processed_skill = self._preprocess_text(skill)
            if not self.collection.find_one(
                {"type": "skill", "value": processed_skill}
            ):
                self.collection.insert_one(
                    {
                        "type": "skill",
                        "value": processed_skill,
                    }
                )
                added_count += 1
            else:
                existing_count += 1

        return {
            "message": f"Added {added_count} new skills, {existing_count} were already present"
        }

    def add_multiple_titles(self, titles: List[str]) -> Dict[str, Any]:
        """Add multiple titles at once"""
        added_count = 0
        existing_count = 0

        for title in titles:
            processed_title = self._preprocess_text(title)
            if not self.collection.find_one(
                {"type": "title", "value": processed_title}
            ):
                self.collection.insert_one(
                    {
                        "type": "title",
                        "value": processed_title,
                    }
                )
                added_count += 1
            else:
                existing_count += 1

        return {
            "message": f"Added {added_count} new titles, {existing_count} were already present"
        }

    def get_all_skills(self) -> List[str]:
        """Get all skills"""
        skills = self.collection.find({"type": "skill"})
        return [skill["value"] for skill in skills]

    def get_all_titles(self) -> List[str]:
        """Get all titles"""
        titles = self.collection.find({"type": "title"})
        return [title["value"] for title in titles]

    def delete_skill(self, skill: str) -> Dict[str, str]:
        """Delete a skill"""
        processed_skill = self._preprocess_text(skill)
        result = self.collection.delete_one({"type": "skill", "value": processed_skill})

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"Skill '{skill}' not found")
        return {"message": f"Skill '{skill}' deleted successfully"}

    def delete_title(self, title: str) -> Dict[str, str]:
        """Delete a title"""
        processed_title = self._preprocess_text(title)
        result = self.collection.delete_one({"type": "title", "value": processed_title})

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"Title '{title}' not found")
        return {"message": f"Title '{title}' deleted successfully"}

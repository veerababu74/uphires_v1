from .rag_application import RAGApplication
from .config import RAGConfig

# Global instance for backward compatibility
rag_app = None


def initialize_rag_app():
    """Initialize the global RAG application instance"""
    global rag_app
    if rag_app is None:
        rag_app = RAGApplication()
    return rag_app


# Enhanced global functions with presets
def ask_resume_question_enhanced(question, mongodb_limit=50, llm_limit=10):
    """Enhanced global function to ask resume questions with configurable limits"""
    global rag_app
    if rag_app is None:
        rag_app = RAGApplication()

    return rag_app.ask_resume_question_with_limits(
        question=question,
        mongodb_retrieval_limit=mongodb_limit,
        llm_context_limit=llm_limit,
    )


def ask_resume_question_preset(question, preset="balanced"):
    """Ask resume questions using predefined performance presets"""
    if preset not in RAGConfig.PERFORMANCE_PRESETS:
        raise ValueError(
            f"Invalid preset. Choose from: {list(RAGConfig.PERFORMANCE_PRESETS.keys())}"
        )

    config = RAGConfig.PERFORMANCE_PRESETS[preset]
    return ask_resume_question_enhanced(
        question, mongodb_limit=config["mongodb_limit"], llm_limit=config["llm_limit"]
    )


def main():
    """Main function with improved interface"""
    try:
        # Initialize RAG application
        rag_app = RAGApplication()
        print("✅ RAG Application initialized successfully")

        while True:
            print("\n🔍 Search Options:")
            print("1. Vector Similarity Search (fast, based on vector scores)")
            print("2. LLM Context Search (detailed analysis with explanations)")
            print("3. Exit")

            choice = input("\nEnter your choice (1-3): ").strip()

            if choice == "3":
                print("Goodbye!")
                break

            query = input("\nEnter your search query: ").strip()

            if choice == "1":
                limit = int(
                    input("Enter number of results to retrieve (default 50): ") or "50"
                )
                result = rag_app.vector_similarity_search(query, limit)
                print(type(result))

                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                    continue

                print(f"\n📊 Found {result['total_found']} results:")
                for i, doc in enumerate(result["results"], 1):
                    print(f"\nRank {i}:")
                    print(f"ID: {doc['_id']}")
                    print(f"Name: {doc['contact_details']['name']}")
                    print(f"Score: {doc['similarity_score']:.4f}")
                    print(f"Skills: {', '.join(doc['skills'])}")
                    print(f"Experience: {doc['total_experience']}")
                    print(f"Current Location: {doc['contact_details']['current_city']}")
                    print("-" * 50)

            elif choice == "2":
                context_size = int(
                    input("Enter number of documents to analyze (default 5): ") or "5"
                )
                result = rag_app.llm_context_search(query, context_size)
                print(type(result))
                print(result)

                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                    continue

                print(f"\n📊 Analyzed {result['total_analyzed']} documents:")
                print(f"Statistics: {result['statistics']}")
                print("Results:")
                print(f"Total Results: {result['total_found']}")
                print(f"Total Analyzed: {result['total_analyzed']}")
                for i, doc in enumerate(result["results"], 1):
                    print(f"\nRank {i}:")
                    print(f"ID: {doc['_id']}")
                    print(f"Name: {doc['contact_details']['name']}")
                    print(f"Relevance Score: {doc['relevance_score']:.4f}")
                    print(f"Match Reason: {doc['match_reason']}")
                    print(f"Skills: {', '.join(doc['skills'])}")
                    print(f"Experience: {doc['total_experience']}")
                    print(f"Current Location: {doc['contact_details']['current_city']}")
                    print("-" * 50)

            else:
                print("❌ Invalid choice. Please try again.")

    except Exception as e:
        print(f"💥 An error occurred: {str(e)}")


if __name__ == "__main__":
    main()

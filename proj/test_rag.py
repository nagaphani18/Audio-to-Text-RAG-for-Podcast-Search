"""
Test script to debug RAG engine initialization issues.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test if all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.rag_engine import RAGEngine, create_rag_engine
        print("✅ RAG engine imports successful")
    except Exception as e:
        print(f"❌ RAG engine import failed: {e}")
        return False
    
    try:
        from src.audio_processor import AudioProcessor
        print("✅ Audio processor import successful")
    except Exception as e:
        print(f"❌ Audio processor import failed: {e}")
        return False
    
    try:
        from src.text_processor import TextProcessor
        print("✅ Text processor import successful")
    except Exception as e:
        print(f"❌ Text processor import failed: {e}")
        return False
    
    try:
        from src.embeddings import EmbeddingManager
        print("✅ Embedding manager import successful")
    except Exception as e:
        print(f"❌ Embedding manager import failed: {e}")
        return False
    
    try:
        from src.utils import load_config
        print("✅ Utils import successful")
    except Exception as e:
        print(f"❌ Utils import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from src.utils import load_config
        config = load_config()
        print(f"✅ Config loaded: {config}")
        return config
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return None

def test_rag_engine():
    """Test RAG engine creation."""
    print("\nTesting RAG engine creation...")
    
    try:
        from src.rag_engine import create_rag_engine
        from src.utils import load_config
        
        config = load_config()
        print(f"Config: {config}")
        
        rag_engine = create_rag_engine(config)
        print("✅ RAG engine created successfully")
        
        # Test basic methods
        stats = rag_engine.get_database_stats()
        print(f"✅ Database stats: {stats}")
        
        return True
    except Exception as e:
        print(f"❌ RAG engine creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🔍 RAG Engine Debug Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Check dependencies.")
        return
    
    # Test config
    config = test_config()
    if not config:
        print("\n❌ Config test failed.")
        return
    
    # Test RAG engine
    if test_rag_engine():
        print("\n✅ All tests passed! RAG engine should work.")
    else:
        print("\n❌ RAG engine test failed. Check the error above.")

if __name__ == "__main__":
    main() 
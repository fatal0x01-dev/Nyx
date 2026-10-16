from memory.rag_store import RAGStore


def test_system_context_persists_by_environment_and_can_be_retrieved(tmp_path):
    db_path = tmp_path / "rag.sqlite3"
    context = {
        "current_directory": "/home/kali/projects/web",
        "home_directory": "/home/kali",
        "user": "kali",
        "hostname": "kali-box",
        "os_kernel": "Linux 6.1 x86_64",
    }

    RAGStore(str(db_path)).save_system_context("kali@192.0.2.8", context)

    reopened = RAGStore(str(db_path))
    assert reopened.get_system_context("kali@192.0.2.8") == context
    assert reopened.get_system_context("local") is None
    assert "/home/kali/projects/web" in reopened.query("current directory web", top_k=1)[0]

import shutil
import os

class FolderManager:
    def __init__(self):
        """Initialisiert den FolderManager mit dem Pfad zum Ordner."""
        self.script_path = os.path.abspath(__file__)
        self.script_dir = os.path.dirname(os.path.dirname(self.script_path))
        self.base_dir = "/Users/markusfreyt/Development/Projects/AI"

    def delete_database(self) -> None:
        """Löscht den gesamten Ordner und dessen Inhalt."""
        if os.path.exists(self.base_dir):
            try:
                shutil.rmtree(f"{self.base_dir}/db")
                print("Ordner erfolgreich gelöscht.")
            except Exception as e:
                print(f"Fehler beim Löschen des Ordners: {e}")

    def get_script_path(self) -> str:
        """Gibt den Pfad des aktuellen Skripts zurück."""
        return self.script_path

    def get_script_dir(self) -> str:
        """Gibt das Verzeichnis des aktuellen Skripts zurück."""
        return self.script_dir

    def get_base_dir(self) -> str:
        """Gibt das Basisverzeichnis (d.h., das Großelternverzeichnis) zurück."""
        return self.base_dir
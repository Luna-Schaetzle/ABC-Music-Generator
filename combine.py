# Dieses Skript kombiniert alle Dateien dataset-1.txt bis dataset-15.txt
# in die Datei dataset-combined.txt, indem es den Inhalt jeder Datei an die Kombinationsdatei anhängt.

def combine_datasets():
    with open("dataset-combined.txt", "a", encoding="utf-8") as outfile:
        for i in range(1, 16):
            filename = f"dataset-{i}.txt"
            try:
                with open(filename, "r", encoding="utf-8") as infile:
                    content = infile.read()
                    outfile.write(content)
                    outfile.write("\n")  # optional: fügt eine neue Zeile zwischen den Dateien hinzu
            except FileNotFoundError:
                print(f"Datei {filename} nicht gefunden.")

if __name__ == "__main__":
    combine_datasets()
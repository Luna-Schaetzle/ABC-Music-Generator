# ASCII Music Generator
### by Luna Schätzle    

## Project Overview

This project explores how music notes can be represented using ASCII characters through **ABC Notation**. ABC Notation is a text-based musical notation system that allows music to be stored and processed as plain text. 

We use datasets containing ABC Notation to train a neural network capable of generating realistic musical sequences.

One such dataset is available at: [ABC Notation Music Dataset](https://abc.sourceforge.net/NMD/).

## The Model

Using these ASCII-based musical notations, we train a machine learning model to generate new ABC Notation sequences. The model is based on an LSTM (Long Short-Term Memory) network, which learns patterns in musical structures to create coherent compositions.

Once trained, the model is saved as a `.h5` file. This file can then be used to generate ABC Notation text, which is subsequently saved as a `.abc` file.

## Converting and Playing Music

1. **Generate ABC Notation**: The trained model produces a sequence in ABC format.
2. **Convert ABC to MIDI**: The ABC notation is converted into a `.mid` (MIDI) file using `abc2midi`.
3. **Playback**: The generated MIDI file can be played back using a built-in player with graphical controls.

## Installation

To set up the project, ensure you have the following dependencies installed:

```bash
pip install tensorflow numpy pygame
```

Additionally, install `abc2midi` to convert ABC notation into MIDI:

- On **Ubuntu**:
  ```bash
  sudo apt install abcmidi
  ```
- On **Windows**:
  Download and install [ABC Music Software](https://abc.sourceforge.net/abcMIDI/) manually.

## Usage

1. Train the model using:
   ```bash
   python train_model.py
   ```
   This will generate a `.h5` model file.

2. Run the graphical user interface (GUI) to generate music:
   ```bash
   python gui.py
   ```

3. Enter a seed text, adjust parameters, generate ABC notation, and listen to the result.

## Features
- **Neural Network-based Music Generation**
- **Graphical Interface for User Interaction**
- **ABC to MIDI Conversion & Playback**
- **Model Training & Saving**

## Contributing
If you would like to contribute, feel free to submit a pull request or open an issue.

## License
This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

# ASCII Music Generator – Wissenschaftliche Dokumentation

## 1. Einleitung

Der **ASCII Music Generator** demonstriert, wie musikalische Informationen mithilfe von Textrepräsentationen und neuronalen Netzen verarbeitet und generiert werden können. Im Mittelpunkt steht die Verwendung der **ABC-Notation**, einem textbasierten Format zur Darstellung von Musiknoten, das sich für die maschinelle Verarbeitung hervorragend eignet. Ziel dieses Projekts ist es, ein Modell zu trainieren, das in der Lage ist, neue musikalische Sequenzen in ABC-Notation zu generieren, diese in MIDI-Dateien umzuwandeln und letztlich abzuspielen.

## 2. Datenakquise und -vorverarbeitung

Die Funktion `load_data` übernimmt das Einlesen eines Datensatzes, der in einem einfachen Textformat (ABC-Notation) vorliegt. Wissenschaftlich betrachtet erfolgt hier eine **Feature-Extraktion**:

- **Zeichensatz-Bestimmung:** Alle eindeutigen Zeichen (Token) des Textes werden ermittelt, was zur Erstellung eines Vokabulars führt.
- **Mapping:** Es wird ein Mapping zwischen den Zeichen und numerischen Indizes erstellt, um den Text in ein Format zu überführen, das von neuronalen Netzen verarbeitet werden kann.
- **Sequenzgenerierung:** Mit Hilfe eines gleitenden Fensters (definiert durch `MAXLEN` und `STEP`) werden Eingabesequenzen und zugehörige Zielzeichen extrahiert. Diese Sequenzen bilden die Basis für das Training.

Diese Vorverarbeitungsschritte entsprechen in vielen Bereichen der natürlichen Sprachverarbeitung (NLP) der Tokenisierung und der Erstellung von Trainingssequenzen.

## 3. Modellarchitektur und Training

Das Modell basiert auf einem rekurrenten neuronalen Netz (RNN) mit LSTM-Zellen, das für die Erfassung langfristiger Abhängigkeiten in sequentiellen Daten geeignet ist:

- **Embedding-Schicht:** Wandelt diskrete Zeichen in kontinuierliche Vektorraumdarstellungen um, was semantische Beziehungen zwischen den Zeichen ermöglicht.
- **LSTM-Schichten:** Zwei hintereinandergeschaltete LSTM-Schichten (mit Dropout zur Regularisierung) erfassen die zeitlichen Muster in den Sequenzen.
- **Dense-Schicht:** Eine abschließende fully-connected Schicht mit Softmax-Aktivierung gibt Wahrscheinlichkeitsverteilungen über das Vokabular aus, sodass das nächste Zeichen auf Basis der gelernten Wahrscheinlichkeiten ausgewählt werden kann.

Die Funktion `build_model` implementiert diese Architektur, und das Training erfolgt über die Funktion `train_and_save_model`, die das Modell auf den vorverarbeiteten Daten trainiert und anschließend als `.h5`-Datei speichert.

## 4. Textgenerierung

Die Funktion `generate_text` implementiert eine probabilistische Textgenerierung. Dabei werden folgende Schritte durchgeführt:

- **Seed-Verwendung:** Ein Startstring (Seed) wird als initiale Eingabe genutzt.
- **Sequenzvorhersage:** Für jede Iteration wird eine Eingabesequenz in das Modell eingespeist, um eine Wahrscheinlichkeitsverteilung über das nächste Zeichen zu erhalten.
- **Sampling:** Mittels Temperaturparameter wird das Sampling-Verhalten gesteuert. Eine niedrigere Temperatur führt zu deterministischeren Ausgaben, während eine höhere Temperatur mehr Varianz und Kreativität einbringt.
- **Sequenz-Update:** Das neu generierte Zeichen wird an den Seed angehängt, wobei das älteste Zeichen entfernt wird, um die Sequenzlänge konstant zu halten.

Diese iterative Methode erlaubt es, lange Sequenzen zu erzeugen, die in musikalischer Hinsicht zusammenhängende Strukturen aufweisen können.

## 5. Konvertierung und Abspielbarkeit

Nach der Generierung erfolgt die Konvertierung der ABC-Notation in ein MIDI-Format:

- **Konvertierung:** Die Funktion `convert_abc_to_midi` speichert den generierten ABC-Text in einer Datei und ruft das externe Tool `abc2midi` auf, um die MIDI-Datei zu erstellen.
- **Abspielbarkeit:** Mittels `pygame` wird die MIDI-Datei geladen und abgespielt. Funktionen wie `play_midi` bieten hierbei eine einfache Möglichkeit, das Ergebnis direkt zu evaluieren.

## 6. Graphische Benutzeroberfläche (GUI)

Das Programm enthält eine umfassende GUI, entwickelt mit Tkinter, die folgende Funktionen bereitstellt:

- **Parameter-Eingabe:** Der Benutzer kann den Seed, die Temperatur und die gewünschte Länge der Ausgabe festlegen.
- **Anzeige:** Der generierte ABC-Text wird in einem Textfeld angezeigt.
- **Dateispeicherung:** Eine Funktion erlaubt das Speichern der generierten Musik in einer `.abc`-Datei.
- **Wiedergabe:** Durch einen integrierten MIDI-Player kann die erzeugte Musik direkt abgespielt werden.

Die GUI vereinfacht die Interaktion mit dem Modell und ermöglicht auch Anwendern ohne tiefergehende Programmierkenntnisse, experimentelle Musikgeneration durchzuführen.

## 7. Zusammenfassung und Ausblick

Dieses Projekt zeigt, wie Techniken der natürlichen Sprachverarbeitung und neuronalen Netze zur Musikgenerierung adaptiert werden können. Neben der praktischen Umsetzung bietet der Ansatz auch ein interessantes Forschungsfeld, in dem weitergehende Fragen zu Kreativität, Variation und Struktur in generierter Musik untersucht werden können.

Für zukünftige Arbeiten bieten sich folgende Erweiterungen an:
- Integration komplexerer Architekturen wie Transformer-Modelle.
- Erweiterte Trainingsdatensätze und erweiterte Vorverarbeitungsstrategien.
- Verbesserte Evaluation der generierten Musik hinsichtlich musikalischer Qualität und Struktur.



import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import pygame

# ----------------------------
# Parameter und Dateinamen
# ----------------------------
DATASET_PATH = "dataset-combined.txt"        # Pfad zu deinem ABC-Datensatz
MODEL_PATH = "ascii_music_model.h5"       # Dateiname für das gespeicherte Modell
MAXLEN = 40                               # Länge der Eingabesequenzen
STEP = 3                                  # Schrittweite bei der Erstellung der Sequenzen

# ----------------------------
# Daten laden und vorbereiten
# ----------------------------
def load_data(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Datensatzlänge: {len(text)} Zeichen")

    # Alle einzigartigen Zeichen
    chars = sorted(list(set(text)))
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for idx, char in enumerate(chars)}
    vocab_size = len(chars)
    print(f"Vokabulargröße: {vocab_size}")

    # Erstellen von Sequenzen und Zielzeichen
    sentences = []
    next_chars = []
    for i in range(0, len(text) - MAXLEN, STEP):
        sentences.append(text[i: i + MAXLEN])
        next_chars.append(text[i + MAXLEN])
    print("Anzahl der Sequenzen:", len(sentences))

    # Numerische Repräsentierung der Daten
    X = np.zeros((len(sentences), MAXLEN), dtype=np.int32)
    y = np.zeros((len(sentences)), dtype=np.int32)
    for i, sentence in enumerate(sentences):
        X[i] = [char_to_idx[char] for char in sentence]
        y[i] = char_to_idx[next_chars[i]]

    return text, chars, char_to_idx, idx_to_char, vocab_size, X, y

# ----------------------------
# Modell erstellen (verbessert)
# ----------------------------
def build_model(vocab_size, maxlen):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=maxlen))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    model.summary()
    return model

# ----------------------------
# Textgenerierung
# ----------------------------
def generate_text(model, seed, char_to_idx, idx_to_char, vocab_size, length=400, temperature=1.0):
    generated = seed
    sentence = seed
    for i in range(length):
        progress = int((i + 1) * 50 / length)
        bar = '[' + '#' * progress + '-' * (50 - progress) + ']'
        print(f"\rGeneriere: {bar} {i + 1}/{length}", end='', flush=True)
        # Vorhersage des nächsten Zeichens
        x_pred = np.zeros((1, MAXLEN), dtype=np.int32)
        for t, char in enumerate(sentence):
            # Wenn der Seed nicht im Vokabular enthalten ist, überspringen
            x_pred[0, t] = char_to_idx.get(char, 0)
        preds = model.predict(x_pred, verbose=0)[0]
        preds = np.log(preds + 1e-8) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        next_index = np.random.choice(range(vocab_size), p=preds)
        next_char = idx_to_char[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char
    return generated

# ----------------------------
# ABC in MIDI konvertieren und abspielen
# ----------------------------
def convert_abc_to_midi(abc_text, abc_filename="generated.abc", midi_filename="generated.mid"):
    # Speichern des ABC-Textes
    with open(abc_filename, "w", encoding="utf-8") as f:
        f.write(abc_text)
    # Aufruf von abc2midi (muss installiert sein)
    try:
        subprocess.run(["abc2midi", abc_filename, "-o", midi_filename], check=True)
        return midi_filename
    except Exception as e:
        messagebox.showerror("Fehler", f"Fehler bei der Konvertierung: {e}")
        return None

def play_midi(midi_file):
    if not os.path.exists(midi_file):
        messagebox.showerror("Fehler", "MIDI-Datei nicht gefunden!")
        return
    # Initialisiere pygame mixer
    pygame.mixer.init()
    try:
        pygame.mixer.music.load(midi_file)
        pygame.mixer.music.play()
    except Exception as e:
        messagebox.showerror("Fehler", f"Fehler beim Abspielen: {e}")

# ----------------------------
# Training und Modell laden
# ----------------------------
def train_and_save_model():
    # Daten laden
    text, chars, char_to_idx, idx_to_char, vocab_size, X, y = load_data(DATASET_PATH)
    # Modell erstellen
    model = build_model(vocab_size, MAXLEN)
    # Training (hier für Demo-Zwecke relativ wenige Epochen)
    model.fit(X, y, batch_size=128, epochs=10)
    # Modell speichern
    model.save(MODEL_PATH)
    messagebox.showinfo("Training", "Modelltraining abgeschlossen und Modell gespeichert!")
    return model, text, chars, char_to_idx, idx_to_char, vocab_size

# ----------------------------
# GUI
# ----------------------------
class AsciiMusicGUI:
    def __init__(self, master):
        self.master = master
        master.title("ASCII-Musik Generator")
        
        # Model und Daten laden oder trainieren
        if os.path.exists(MODEL_PATH):
            try:
                self.model = load_model(MODEL_PATH)
                # Lade zusätzlich die Trainingsdaten, um Mapping zu erhalten
                _, self.chars, self.char_to_idx, self.idx_to_char, self.vocab_size, self.X, self.y = load_data(DATASET_PATH)
                with open(DATASET_PATH, "r", encoding="utf-8") as f:
                    self.full_text = f.read()
                messagebox.showinfo("Modell geladen", "Vorhandenes Modell wurde geladen.")
            except Exception as e:
                messagebox.showerror("Fehler", f"Modell konnte nicht geladen werden: {e}")
                self.model = None
        else:
            # Falls kein Modell vorhanden, Training starten
            self.model, self.full_text, self.chars, self.char_to_idx, self.idx_to_char, self.vocab_size = train_and_save_model()
        
        # GUI-Elemente
        tk.Label(master, text="Start-Seed (mind. 40 Zeichen):").pack()
        self.seed_entry = tk.Entry(master, width=60)
        self.seed_entry.insert(0, self.full_text[:MAXLEN])
        self.seed_entry.pack()
        
        tk.Label(master, text="Temperatur (z. B. 1.0):").pack()
        self.temp_entry = tk.Entry(master, width=10)
        self.temp_entry.insert(0, "1.0")
        self.temp_entry.pack()
        
        tk.Label(master, text="Länge der Ausgabe:").pack()
        self.length_entry = tk.Entry(master, width=10)
        self.length_entry.insert(0, "400")
        self.length_entry.pack()
        
        self.generate_button = tk.Button(master, text="Generieren", command=self.generate_and_display)
        self.generate_button.pack()
        
        self.text_output = tk.Text(master, height=20, width=80)
        self.text_output.pack()
        
        frame = tk.Frame(master)
        frame.pack(pady=5)
        self.save_button = tk.Button(frame, text="Speichern als .abc", command=self.save_to_file)
        self.save_button.grid(row=0, column=0, padx=5)
        self.play_button = tk.Button(frame, text="Abspielen", command=self.generate_and_play)
        self.play_button.grid(row=0, column=1, padx=5)
    
    def generate_and_display(self):
        seed = self.seed_entry.get()
        if len(seed) < MAXLEN:
            messagebox.showerror("Fehler", f"Der Seed muss mindestens {MAXLEN} Zeichen lang sein.")
            return
        try:
            temp = float(self.temp_entry.get())
            length = int(self.length_entry.get())
        except ValueError:
            messagebox.showerror("Fehler", "Bitte gültige numerische Werte für Temperatur und Länge eingeben.")
            return
        generated_text = generate_text(self.model, seed, self.char_to_idx, self.idx_to_char, self.vocab_size, length, temp)
        self.generated_text = generated_text  # speichern für spätere Verwendung
        self.text_output.delete("1.0", tk.END)
        self.text_output.insert(tk.END, generated_text)
    
    def save_to_file(self):
        text_data = self.text_output.get("1.0", tk.END)
        file_path = filedialog.asksaveasfilename(defaultextension=".abc", filetypes=[("ABC Notation", "*.abc")])
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text_data)
            messagebox.showinfo("Gespeichert", "Datei erfolgreich gespeichert!")
    
    def generate_and_play(self):
        # Zuerst generieren
        self.generate_and_display()
        if hasattr(self, "generated_text"):
            midi_file = convert_abc_to_midi(self.generated_text)
            if midi_file:
                play_midi(midi_file)

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # Initialisiere pygame (für MIDI-Abspielung)
    pygame.init()
    
    root = tk.Tk()
    app = AsciiMusicGUI(root)
    root.mainloop()


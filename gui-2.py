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
DATASET_PATH = "dataset-1.txt"        # Pfad zu deinem ABC-Datensatz
MODEL_PATH = "ascii_music_model.h5"     # Gespeichertes Modell
MAXLEN = 40  # Länge der Eingabesequenzen
STEP = 3      # Schrittweite bei der Erstellung der Sequenzen

# ----------------------------
# MIDI Player-Klasse
# ----------------------------
class MidiPlayer:
    def __init__(self):
        pygame.mixer.init()
        self.is_paused = False
    
    def play(self, midi_file):
        if not os.path.exists(midi_file):
            messagebox.showerror("Fehler", "MIDI-Datei nicht gefunden!")
            return
        pygame.mixer.music.load(midi_file)
        pygame.mixer.music.play()
    
    def pause(self):
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.pause()
            self.is_paused = True
    
    def resume(self):
        if self.is_paused:
            pygame.mixer.music.unpause()
            self.is_paused = False
    
    def stop(self):
        pygame.mixer.music.stop()

player = MidiPlayer()

# ----------------------------
# GUI für Musik-Generierung und Abspielen
# ----------------------------
class AsciiMusicGUI:
    def __init__(self, master):
        self.master = master
        master.title("ASCII-Musik Generator")
        
        tk.Label(master, text="Start-Seed (mind. 40 Zeichen):").pack()
        self.seed_entry = tk.Entry(master, width=60)
        self.seed_entry.pack()
        
        self.generate_button = tk.Button(master, text="Generieren", command=self.generate_text)
        self.generate_button.pack()
        
        self.text_output = tk.Text(master, height=10, width=80)
        self.text_output.pack()
        
        # Steuerbuttons für MIDI
        control_frame = tk.Frame(master)
        control_frame.pack()
        
        self.play_button = tk.Button(control_frame, text="▶ Play", command=self.play_midi)
        self.play_button.grid(row=0, column=0, padx=5)
        
        self.pause_button = tk.Button(control_frame, text="⏸ Pause", command=player.pause)
        self.pause_button.grid(row=0, column=1, padx=5)
        
        self.resume_button = tk.Button(control_frame, text="▶ Resume", command=player.resume)
        self.resume_button.grid(row=0, column=2, padx=5)
        
        self.stop_button = tk.Button(control_frame, text="⏹ Stop", command=player.stop)
        self.stop_button.grid(row=0, column=3, padx=5)
        
        self.generated_text = ""
        
    def generate_text(self):
        seed = self.seed_entry.get()
        if len(seed) < MAXLEN:
            messagebox.showerror("Fehler", f"Der Seed muss mindestens {MAXLEN} Zeichen lang sein.")
            return
        
        # Fiktive Textgenerierung (Hier dein Modell einbinden)
        self.generated_text = "X:1\nT:Generated Tune\nM:4/4\nK:C\nCDEF GABc|cBAG FEDC|..."
        self.text_output.delete("1.0", tk.END)
        self.text_output.insert(tk.END, self.generated_text)
    
    def play_midi(self):
        if not self.generated_text:
            messagebox.showerror("Fehler", "Bitte erst Musik generieren!")
            return
        
        midi_file = "generated.mid"
        with open("generated.abc", "w") as f:
            f.write(self.generated_text)
        
        subprocess.run(["abc2midi", "generated.abc", "-o", midi_file], check=True)
        player.play(midi_file)

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    pygame.init()
    root = tk.Tk()
    app = AsciiMusicGUI(root)
    root.mainloop()


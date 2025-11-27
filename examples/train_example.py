from vocalid.trainer import VoiceTrainer
from glob import glob

positive = glob("data/my_voices/*.wav")
negative = glob("data/other_voices/*.wav")

trainer = VoiceTrainer()
trainer.train(positive, negative, save_path="my_model.pkl")
   
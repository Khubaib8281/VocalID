import argparse
import os
from glob import glob
from .trainer import VoiceTrainer
from .verifier import VoiceVerifier
from .audio_utils import record_audio

def main():
    parser = argparse.ArgumentParser(description="Voice Verifier CLI")
    sub = parser.add_subparsers(dest="commands")

    # training command
    train = sub.add_parser("train", help="Train a voice authentication model")
    train.add_argument("--positive", required=True, help = "Folder with your voice samples")
    train.add_argument("--negative", required=True, help = "Folder with other voices")
    train.add_argument("--output", default="voice_auth.pkl", help="path to save model")

    # evaluate the model
    evaluate = sub.add_parser("evaluate", help = "Evaluate the trained model")
    evaluate.add_argument("--model", required=True, help= "Path to trained model")
    evaluate.add_argument("--positive", required=True, help = "Folder with your voice samples")
    evaluate.add_argument("--negative", required=True, help= "Folder with negative/random samples")

    # verify file command
    verify = sub.add_parser("verify", help = "Verify a voice file")
    verify.add_argument("file", help="path to .wav voice file to verify")
    verify.add_argument("--model", default="voice_auth.pkl", help="trained model path")

    # live verification command
    live = sub.add_parser("live", help = "Live microphone verification")
    live.add_argument("--model", default="voice_auth.pkl", help= "trained model path")
    live.add_argument("--seconds", type=int, default=4, help= "Recording duration in seconds")

    args = parser.parse_args()

    if args.commands == "train":
        pos_files = glob(os.path.join(args.positive, "*.wav"))
        neg_files = glob(os.path.join(args.negative, "*.wav"))
        trainer = VoiceTrainer()
        trainer.train(pos_files, neg_files, args.output)
        print(f"Model saved to {args.output}")

    elif args.commands == "evaluate":
        pos_files = glob(os.path.join(args.positive, "*.wav"))
        neg_files = glob(os.path.join(args.negative, "*.wav"))

        trainer = VoiceTrainer()
        trainer.load(args.model)
        # X, y = trainer.prepare_features(pos_files, neg_files)
        results = trainer.evaluate(pos_files, neg_files)

        print("\n===== Evaluation Results =====")
        print("\nAccuracy", round(results["accuracy"], 4))
        print("\nClassification Report:\n")
        print(results["report"])

    elif args.commands == "verify":
        verifier = VoiceVerifier(args.model)
        ok, score = verifier.verify_file(args.file)
        print(f"Verified: {ok}, Score: {score:.2f}")

    elif args.commands == "live":
        verifier = VoiceVerifier(args.model)
        try:
            audio_tensor = record_audio(args.seconds)
        except RuntimeError as e:
            print(str(e))
            return

        ok, score = verifier.verify_array(audio_tensor)
        print(f"Verified: {ok}, Score: {score:.2f}")

    else:
        parser.print_help()
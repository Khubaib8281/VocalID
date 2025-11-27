from vocalid.verifier import VoiceVerifier

verifier = VoiceVerifier("my_model.pkl")

# Verify file
ok, score = verifier.verify_file("test.wav")
print(ok, score)

# Verify live microphone
ok, score = verifier.verify_live(seconds=4)
print("Live verification:", ok, score)
   
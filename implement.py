import preprocess
import utils
import ENB0

# Define where raw audio is stored
hp_wd = 'F:\\WII_Whale\\HP6\\0093\\'
# Define where the model is stored
model = 'F:\\WII_Whale\\enb0\\enb0_4_2.keras'

# Process the audio
preprocess.audio_process(td=hp_wd,sf=5000)
# Create 2 second clips
preprocess.few_second_clips(td=hp_wd+'filtered\\', clip=2)
# Create spectrograms
preprocess.spec_process(td=hp_wd+'filtered\\2s_clip\\', mid=70)
# Predict labels for spectrograms using model_1
ENB0.predict_unseen(model= model, td = hp_wd, score_threshold =0.5, tag='freeze_enb0_4_2')
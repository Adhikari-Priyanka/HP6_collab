import preprocess
import utils

# Define where raw audio is stored
hp_wd = 'F:\\WII_Whale\\HP6\\0093\\'

# Process the audio
preprocess.audio_process(td=hp_wd,sf=5000)
# Create 2 second clips
preprocess.few_second_clips(td=hp_wd+'filtered\\', clip=2)
# Create spectrograms
preprocess.spec_process(td=hp_wd+'filtered\\2s_clip\\', mid=70)
# Generate list of file names
utils.filename_to_csv(td=hp_wd+'filtered\\2s_clip\\specs\\', format='.png')
## Assign 'Yes' or 'No' label to each file
# Move spectrograms into different folders by label
utils.move_to_label(source=hp_wd+'filtered\\2s_clip\\specs\\', target=hp_wd+'filtered\\2s_clip\\specs\\',
                    format= '.png', df= 'F:\\WII_Whale\\annotation_csvs\\0093_first10min.csv')
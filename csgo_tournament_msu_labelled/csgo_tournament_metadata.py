import platform
import sys
path_delim = '\\' if platform.system() == 'Windows' else '/'

NUMBER_OF_PLAYERS = 10
MIN_ROUND_DURATION = 10

MATCH_LEN={
    1:4,
    2:27,
    3:37,
    4:31}


EMOTION_WITH_KEYS={
  1:'интерес',
  2:'радость',
  3:'удивление',
  4:'горе',
  5:'гнев',
  6:'отвращение',
  7:'презрение',
  8:'страх',
  9:'стыд',
 10:'вина'
}

SAMPLING_RATE = 22050
PCS_LEN_SEC = 3

def get_players(n_match):
    players=['incr0ss','Softcore','humllet','faceitkirjke','SL4VAMARL0W']
    if n_match in [1,2]:
        players+=['___Tox1c___','giena1337','TheDefenderr','HOoL1GAN_','DENJKEZOR666'] #- VTB
    elif n_match==3:
        players+=['zhenn--','riddle','savagekx','Ka1n___','_SEGA'] #- GBCB
    elif n_match==4:
        players+=['zhenn--','riddle','savagekx','Ka1n___','RubinskiyRV'] #- GBCB

    return players


parsed_demo_filename = ['3248aa5e-b344-40f5-8f83-4988a3b7141b_de_vertigo_128.csv',
                        '83fdd578-cb07-4c86-abb1-304cb0328b78_de_overpass_128.csv',
                        '3e9849b7-304d-4017-96bc-41e7f0ce6a4e_de_vertigo_128.csv',
                        'b504a2f2-82b6-4385-b0f3-ef9b88949655_de_mirage_128.csv']

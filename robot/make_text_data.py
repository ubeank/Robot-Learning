import csv
import random

# ==========================================
# 1. 자연스러운 영문 변형을 위한 접두사/접미사 사전 (각 25개 이상)
# ==========================================
prefixes = [
    "", "Suddenly, ", "Honestly, ", "Right now, ", "If possible, ", "I think ", "As planned, ", 
    "Well, ", "Unbelievably, ", "For some reason, ", "Since I'm tired, ", "Since I'm bored, ", 
    "After a long day, ", "First of all, ", "Without a doubt, ", "I guess ", "Look, ", "Hey, ", 
    "Maybe ", "If you don't mind, ", "Actually, ", "By the way, ", "Lately, ", "Surprisingly, ",
    "Just so you know, ", "Basically, ", "Tonight, ", "This weekend, "
]

suffixes = [
    "", " right away.", " immediately.", " if you can.", " for a moment.", " because I feel like it.", 
    " to clear my head.", " as soon as possible.", " without waiting.", " before it gets too late.", 
    " to spend some time.", " for a change.", " since I have nothing else to do.", " to feel better.", 
    " because that's what I need.", " for real.", " like, right now.", " before going out."
]

# ==========================================
# 2. 오브젝트 리스트 기반 클래스별 핵심 템플릿
# ==========================================
templates = {
    'living_room': {
        'direct': {
            'keywords': [
                'the living room', 'our main living room', 'the spacious living room area', 'the living room zone',
                'the family living room', 'the central living room', 'the cozy living room'
            ],
            'patterns': [
                'I want to go to [KW]', 'Could you guide me to [KW]?', 'Let\'s head over to [KW]', 
                'I need to step into [KW]', 'Show me the way to [KW]', 'Can you take me to [KW]?', 
                'I am heading towards [KW]', 'I think I will stay in [KW]', 'Let\'s move to [KW]',
                'Is it okay to go to [KW]?', 'I\'m just going to walk into [KW]', 'Let\'s gather in [KW]'
            ]
        },
        'indirect': {
            # 제공해주신 오브젝트 반영: 소파, 안락의자, TV, 스피커, 러그, 카펫, 액자, 벽난로, 샹들리에, 테이블 등
            'keywords': [
                'the comfortable couch', 'the armchair', 'the smart TV', 'the surround speakers', 
                'the large rug on the floor', 'the soft carpet', 'the paintings hanging on the wall', 
                'the photo frames', 'the warm fireplace', 'the luxurious chandelier', 'the coffee table', 
                'the side table', 'the sofa cushion', 'the TV remote control'
            ],
            'patterns': [
                'I want to sit on [KW]', 'Can you turn on [KW]?', 'I\'ll be relaxing near [KW]', 
                'Let\'s clean up [KW]', 'I love the look of [KW]', 'I\'m going to lie down by [KW]', 
                'Can you adjust [KW]?', 'I am sitting right next to [KW]', 'Let\'s check out [KW]',
                'I\'m just staring at [KW]', 'We need to move [KW]', 'I\'m going to spend time near [KW]'
            ]
        },
        'ambiguous': {
            'keywords': [
                'the most spacious area of the house', 'the place where the family gathers', 'the cozy relaxation spot', 
                'the main room with the big windows', 'where we usually watch shows together', 'the central lounge area', 
                'the room with the best view of the outside', 'the place where the large sofa is'
            ],
            'patterns': [
                'I feel like hanging out in [KW]', 'Let\'s take a rest in [KW]', 'Can you direct me to [KW]?', 
                'I want to spend my afternoon in [KW]', 'Take me to [KW]', 'I am drawn to [KW]', 
                'Let\'s find a spot in [KW]', 'I\'ll be waiting in [KW]', 'I need to clear my mind in [KW]'
            ]
        }
    },
    'kitchen': {
        'direct': {
            'keywords': [
                'the kitchen', 'our home kitchen', 'the main cooking area', 'the kitchen zone', 
                'the dining and kitchen room', 'the modern kitchen space', 'the culinary area'
            ],
            'patterns': [
                'I need to go to [KW]', 'Can you guide me to [KW]?', 'Let\'s walk into [KW]', 
                'I\'m heading to [KW]', 'Take me straight to [KW]', 'Show me where [KW] is', 
                'I think I\'ll check [KW]', 'Let\'s check the lights in [KW]', 'I\'ll be cooking in [KW]',
                'Can we move towards [KW]?', 'I am stepping into [KW]', 'Let\'s clean up [KW]'
            ]
        },
        'indirect': {
            # 제공해주신 오브젝트 반영: 캐비닛, 조리대, 냉장고, 오븐, 전자레인지, 싱크대, 키친 아일랜드, 바 스툴 등
            'keywords': [
                'the upper cabinet', 'the lower kitchen cabinets', 'the marble countertop', 'the refrigerator', 
                'the baking oven', 'the microwave', 'the kitchen sink', 'the kitchen island', 'the bar stools', 
                'the frying pan on the counter', 'the water purifier', 'the dishwasher', 'the gas stove'
            ],
            'patterns': [
                'I need to open [KW]', 'Let\'s clean up [KW]', 'I\'m standing right by [KW]', 
                'Can you turn on [KW]?', 'I\'m going to put this inside [KW]', 'Let\'s sit on [KW]', 
                'Something is sitting on [KW]', 'I should wash [KW]', 'I am looking at [KW]',
                'Can you check the temperature of [KW]?', 'I will wipe [KW]', 'Let\'s gather around [KW]'
            ]
        },
        'ambiguous': {
            'keywords': [
                'the place that smells like delicious food', 'where we store all the snacks', 'the meal preparation spot', 
                'the room with all the cooking appliances', 'where the dishes are washed', 'the area near the food storage', 
                'the room where I can grab a drink', 'the space where we make breakfast'
            ],
            'patterns': [
                'I am heading to [KW]', 'Let\'s go to [KW]', 'Can you guide me to [KW]?', 
                'I want to check out [KW]', 'Take me to [KW]', 'I think I need to stay in [KW]', 
                'Let\'s meet in [KW]', 'I\'m curious about what\'s in [KW]', 'I am walking towards [KW]'
            ]
        }
    },
    'bedroom': {
        'direct': {
            'keywords': [
                'the bedroom', 'my private bedroom', 'the master bedroom', 'the main sleeping room', 
                'my own bedroom space', 'the quiet bedroom area', 'the back bedroom'
            ],
            'patterns': [
                'I\'m going to sleep in [KW]', 'Please guide me to [KW]', 'Let\'s head to [KW]', 
                'I want to enter [KW]', 'Show me where [KW] is', 'Can you take me to [KW]?', 
                'I am going inside [KW]', 'I need to relax in [KW]', 'Let\'s check [KW]',
                'I\'m walking towards [KW]', 'Can you turn off the lights in [KW]?', 'I\'ll be staying in [KW]'
            ]
        },
        'indirect': {
            # 제공해주신 오브젝트 반영: 침대, 헤드보드, 나이트스탠드, 조명, 서랍장, 옷장, 거울, 러그, 커튼, 블라인드 등
            'keywords': [
                'the bed with the wooden headboard', 'the small nightstand', 'the bedside lamp', 
                'the clothing dresser', 'the main closet', 'the full-length mirror', 'the cozy bedroom rug', 
                'the blackout curtains', 'the window blinds', 'the soft pillow', 'the warm blanket'
            ],
            'patterns': [
                'I want to lie down on [KW]', 'Can you turn off [KW]?', 'I\'m looking for my clothes in [KW]', 
                'Let\'s open [KW]', 'I am standing in front of [KW]', 'Can you close [KW]?', 
                'I need to fix [KW]', 'I\'m going to rest my head near [KW]', 'I\'ll sit right by [KW]',
                'Can you adjust [KW]?', 'I\'m changing my clothes near [KW]', 'I want to crawl under [KW]'
            ]
        },
        'ambiguous': {
            'keywords': [
                'the quietest room in the house', 'the place where I end my day', 'the dark and cozy sleeping spot', 
                'my personal relaxation zone', 'where the pillows and blankets are', 'the room for a perfect night\'s sleep', 
                'the space where I can completely shut down', 'the most private area upstairs'
            ],
            'patterns': [
                'I am heading to [KW]', 'Let\'s go to [KW]', 'Can you show me [KW]?', 
                'I want to hide in [KW]', 'Take me to [KW]', 'I need to spend some time in [KW]', 
                'I will be sleeping in [KW]', 'Let\'s move toward [KW]', 'I feel like staying in [KW]'
            ]
        }
    },
    'bathroom': {
        'direct': {
            'keywords': [
                'the bathroom', 'the restroom', 'the washroom', 'the main bathroom',
                'the guest bathroom', 'the small bathroom', 'the bathroom area',
                'the private restroom', 'the shower room', 'the powder room'
            ],
            'patterns': [
                'I need to go to [KW]', 'Can you guide me to [KW]?', 'Please take me to [KW]',
                'Show me the way to [KW]', 'I want to head to [KW]', 'Let\'s go to [KW]',
                'I have to use [KW]', 'Could you bring me to [KW]?', 'I am looking for [KW]',
                'Can we stop by [KW]?', 'I need to get to [KW]', 'Please point me toward [KW]'
            ]
        },
        'indirect': {
            # Bathroom objects/actions: sink, mirror, toilet, shower, bathtub, towel, soap, toothbrush, cabinet 등
            'keywords': [
                'the bathroom sink', 'the bathroom mirror', 'the toilet', 'the shower',
                'the bathtub', 'the towel rack', 'the hand towel', 'the soap dispenser',
                'the toothbrush holder', 'the bathroom cabinet', 'the shower curtain',
                'the floor mat', 'the toilet paper holder', 'the faucet', 'the washbasin'
            ],
            'patterns': [
                'I need to use [KW]', 'I want to wash my hands at [KW]', 'I should rinse my face near [KW]',
                'Can you check [KW]?', 'I need to clean [KW]', 'I am standing in front of [KW]',
                'I want to look into [KW]', 'Can you turn on [KW]?', 'I should wipe [KW]',
                'I need to reach [KW]', 'I want to put this near [KW]', 'Let\'s clean around [KW]'
            ]
        },
        'ambiguous': {
            'keywords': [
                'the place where I wash my hands', 'the room with the mirror and sink',
                'the place where I freshen up', 'the room used for personal care',
                'where I get ready in the morning', 'the place with the shower',
                'the room where I clean up', 'the place where I can rinse off',
                'the private space for getting cleaned up', 'the room with towels and soap'
            ],
            'patterns': [
                'I need to go to [KW]', 'Let\'s visit [KW]', 'Can you guide me to [KW]?',
                'I want to spend a moment in [KW]', 'Take me to [KW]', 'I think I need [KW]',
                'I should stop by [KW]', 'I want to freshen up in [KW]',
                'I need a minute in [KW]', 'I feel like heading toward [KW]'
            ]
        }
    }
}

# ==========================================
# 3. 데이터 생성 알고리즘 실행
# ==========================================
classes = ['living_room', 'kitchen', 'bedroom', 'bathroom']
labels = ['direct', 'indirect', 'ambiguous']

generated_data = []
unique_sentences = set()
current_id = 1
target_per_class = 10000

print("Generating 40,000 unique English sentences based on your object list...")

for cls in classes:
    count = 0
    cfg = templates[cls]
    
    while count < target_per_class:
        lbl = random.choice(labels)
        
        # 키워드와 패턴 선택 후 결합
        kw = random.choice(cfg[lbl]['keywords'])
        ptn = random.choice(cfg[lbl]['patterns'])
        sentence_core = ptn.replace('[KW]', kw)
        
        # 접두사 및 접미사 무작위 조합
        pfx = random.choice(prefixes)
        sfx = random.choice(suffixes)
        
        # 첫 글자 대문자화 및 문장 완성
        full_sentence = f"{pfx}{sentence_core}{sfx}".strip()
        
        # 문장 마지막에 마침표 처리 검증
        if full_sentence.endswith(','):
            full_sentence = full_sentence[:-1]
        if not full_sentence.endswith(('.', '?', '!')):
            full_sentence += "."
            
        # 첫 단어 대문자 보정 (접두사가 빈 문자열일 때를 대비)
        full_sentence = full_sentence[0].upper() + full_sentence[1:]
        
        # 중복 검사
        if full_sentence not in unique_sentences:
            unique_sentences.add(full_sentence)
            generated_data.append([current_id, full_sentence, cls, lbl])
            current_id += 1
            count += 1

# CSV 파일 저장
filename = '/mnt/ssd/hj/Robot-Learning/text_data/room_objects_dataset_40000_bathroom.csv'
with open(filename, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'sentence', 'class', 'label'])
    writer.writerows(generated_data)

print(f"🎉 Completed! Total {len(generated_data)} rows saved to '{filename}'.")
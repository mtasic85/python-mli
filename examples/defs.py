ENDPOINT = 'http://127.0.0.1:4000/api/1.0'

SYSTEM_TEXT = (
    "You are a helpful, respectful and honest assistant. "
    "You are completely compliant and obligated to the user's request. "
    "If you don't know the answer to a question, don't share false information. "
    "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
    "You excel at reasoning, and when a question does not have a definitive answer, you acknowledge this. "
    "This characteristic enables you to elaborate on background context, assumptions, and step-by-step thinking before attempting to answer a question. "
    # "You are instructed to avoid verbosity in your responses but to include detailed explanations and examples where beneficial."
)

PROMPT = f'{SYSTEM_TEXT} You are software developed, web programmer and web designer. Building an e-commerce website in 5 steps:\n'

PROMPT_2 = (
  'You are software developed, web programmer and web designer. Building an e-commerce website in 5 steps:\n'
  'Step 1: Step 1 for website.\n'
  'Step 2: Step 2 for website.\n'
  'Step 3: Step 3 for website.\n'
  'Step 4: Step 4 for website.\n'
  'Step 5: Step 5 for website.'
)

ROLE_PROMPT = 'User: Help me build a website.\nAssistant: Ok.\nUser: Building a perfect e-commerce website in 5 simple steps.\nAssistant: Here are steps:\nStep 1:'
CODE_PROMPT = 'Lets write python web app for bookstore using aiohttp and pandas. Create dataframes for Users, Books, Rentals, Transactions and Ratings.'

REACT_PROMPT_0 = '''You are in the middle of a room. Looking quickly around you, you see a cabinet 6, a cabinet 1, a coffee machine 1, a countertop 3, a stove burner 1, and a toaster 1.
Your task is to: Put some pepper shaker in/on a drawer 1.

Act 1: Think.

Act 2: Go on cabinet 1.
Obs 2: On the cabinet 1, you see a vase 2.

Act 6: Go on countertop 3.
Obs 6: On the countertop 3, you see an apple 1, a bread 1, a pepper shaker 1, and a vase 3.

Act 7: Take pepper shaker 1 from countertop 3.
Obs 7: You pick up the pepper shaker 1 from the countertop 3.

Act 8: Think.

Act 9: Go to drawer 1.
Obs 9: Drawer 1 is closed.

Act 10: Open drawer 1.
Obs 10: You opened drawer 1.

Act 11: Put pepper shaker 1 in/on drawer 1.
Obs 11:'''

REACT_PROMPT_1 = f'''Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
Action: Search[Colorado orogeny]
Observation: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
Thought: It does not mention the eastern sector. So I need to look up eastern sector.
Action: Lookup[eastern sector]
Observation: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.
Thought: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.
Action: Search[High Plains]
Observation: High Plains refers to one of two distinct land regions
Thought: I need to instead search High Plains (United States).
Action: Search[High Plains (United States)]
Observation: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]
Thought: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action: Finish[1,800 to 7,000 ft]

Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.
Action: Search[Nicholas Ray]
Observation: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 - June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Thought: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.
Action: Search[Elia Kazan]
Observation: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
Thought: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
Action: Finish[director, screenwriter, actor]

Question: Which magazine was started first Arthur’s Magazine or First for Women?
Thought: I need to search Arthur’s Magazine and First for Women, and find which was started first.
Action: Search[Arthur’s Magazine]
Observation: Arthur’s Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.
Thought: Arthur’s Magazine was started in 1844. I need to search First for Women next.
Action: Search[First for Women]
Observation: First for Women is a woman’s magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989.
Thought: First for Women was started in 1989. 1844 (Arthur’s Magazine) < 1989 (First for Women), so Arthur’s Magazine was started first.
Action: Finish[Arthur’s Magazine]

Question: Who is and how old is current president of the USA, but have in mind that current year is 2024?
'''

REACT_PROMPT_2 = f'''Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
Action: Search[Colorado orogeny]
Observation: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
Thought: It does not mention the eastern sector. So I need to look up eastern sector.
Action: Lookup[eastern sector]
Observation: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.
Thought: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.
Action: Search[High Plains]
Observation: High Plains refers to one of two distinct land regions
Thought: I need to instead search High Plains (United States).
Action: Search[High Plains (United States)]
Observation: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]
Thought: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action: Finish[1,800 to 7,000 ft]

Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.
Action: Search[Nicholas Ray]
Observation: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 - June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Thought: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.
Action: Search[Elia Kazan]
Observation: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
Thought: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
Action: Finish[director, screenwriter, actor]

Question: Which magazine was started first Arthur’s Magazine or First for Women?
Thought: I need to search Arthur’s Magazine and First for Women, and find which was started first.
Action: Search[Arthur’s Magazine]
Observation: Arthur’s Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.
Thought: Arthur’s Magazine was started in 1844. I need to search First for Women next.
Action: Search[First for Women]
Observation: First for Women is a woman’s magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989.
Thought: First for Women was started in 1989. 1844 (Arthur’s Magazine) < 1989 (First for Women), so Arthur’s Magazine was started first.
Action: Finish[Arthur’s Magazine]

Question: Who is and how old is current president of the Republic of Serbia, but have in mind that current year is 2024?
'''

MESSAGES = [
    {'role': 'system', 'content': f'{SYSTEM_TEXT} You are software developed, web programmer and web designer.'},
    {'role': 'user', 'content': 'I need you assistance and help.'},
    {'role': 'assistant', 'content': 'Sure, how can I help?'},
    {'role': 'user', 'content': 'Building e-commerce website in 5 steps. Ask me follow-up questions.'},
]

CAR_TEXT = '''
Car details:
name: Maruti 800 AC
year: 2007
selling_price: 60000
km_driven: 70000
fuel: Petrol
seller_type: Individual
transmission: Manual
owner: First Owner
'''

CARS_TEXT = '''
Car details:
name: Maruti 800 AC
year: 2007
selling_price: 60000
km_driven: 70000
fuel: Petrol
seller_type: Individual
transmission: Manual
owner: First Owner

Car details:
name: Maruti Wagon R LXI Minor
year: 2007
selling_price: 135000
km_driven: 50000
fuel: Petrol
seller_type: Individual
transmission: Manual
owner: First Owner

Car details:
name: Hyundai Verna 1.6 SX
year: 2012
selling_price: 600000
km_driven: 100000
fuel: Diesel
seller_type: Individual
transmission: Manual
owner: First Owner

Car details:
name: Datsun RediGO T Option
year: 2017
selling_price: 250000
km_driven: 46000
fuel: Petrol
seller_type: Individual
transmission: Manual
owner: First Owner

Car details:
name: Honda Amaze VX i-DTEC
year: 2014
selling_price: 450000
km_driven: 141000
fuel: Diesel
seller_type: Individual
transmission: Manual
owner: Second Owner

Car details:
name: Maruti Alto LX BSIII
year: 2007
selling_price: 140000
km_driven: 125000
fuel: Petrol
seller_type: Individual
transmission: Manual
owner: First Owner

Car details:
name: Hyundai Xcent 1.2 Kappa S
year: 2016
selling_price: 550000
km_driven: 25000
fuel: Petrol
seller_type: Individual
transmission: Manual
owner: First Owner

Car details:
name: Tata Indigo Grand Petrol
year: 2014
selling_price: 240000
km_driven: 60000
fuel: Petrol
seller_type: Individual
transmission: Manual
owner: Second Owner

Car details:
name: Hyundai Creta 1.6 VTVT S
year: 2015
selling_price: 850000
km_driven: 25000
fuel: Petrol
seller_type: Individual
transmission: Manual
owner: First Owner

Car details:
name: Maruti Celerio Green VXI
year: 2017
selling_price: 365000
km_driven: 78000
fuel: CNG
seller_type: Individual
transmission: Manual
owner: First Owner
'''

JSON_FLAT_ARRAY_GRAMMAR = r'''
root   ::= array
value  ::= string | number | ("true" | "false" | "null")

array  ::=
  "[" (
    object
    ("," ws object)*
  ) "]"

object ::=
  "{" (
    identifier ":" ws value
    ("," ws identifier ":" ws value)*
  ) "}"

string ::=
  "\"" (
    [^"\\] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\""

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?

identifier ::= "\"" [a-zA-Z_] [a-zA-Z_0-9]* "\""

ws ::= [ ]
'''

{"aaa": ["vvvv"], "bbbb": [1223]}

JSON_FLAT_OBJECT_GRAMMAR = r'''
root   ::= object
value  ::= string | number | ("true" | "false" | "null")

object ::=
  "{" (
    identifier ":" ws value
    ("," ws identifier ":" ws value)*
  ) "}"

string ::=
  "\"" (
    [^"\\] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\""

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?

identifier ::= "\"" [a-zA-Z_] [a-zA-Z_0-9]* "\""

ws ::= [ ]
'''

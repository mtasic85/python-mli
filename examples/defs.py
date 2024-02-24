
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

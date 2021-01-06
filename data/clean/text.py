import numpy as np
import inflect
import re 
import tensorflow as tf

_inflect = inflect.engine()
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')

_abbreviations_list = [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]
# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in _abbreviations_list]



def _collapse_whitespace(text):
  return text.replace(" ", "")

def _lowercase(text):
  return text.lower()

def _remove_commas(text):
  return text.replace(',', '')

def _remove_symbols(text):
  text = re.sub(r'[^a-zA-Z0-9 \'.,]', '', text)
  return text



def _expand_decimal_point(text):
  return text.group(1).replace('.', ' point ')

def _expand_dollars(text):
  match = text.group(1)
  parts = match.split('.')
  if len(parts) > 2:
    return match + ' dollars'  # Unexpected format
  dollars = int(parts[0]) if parts[0] else 0
  cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
  if dollars and cents:
    dollar_unit = 'dollar' if dollars == 1 else 'dollars'
    cent_unit = 'cent' if cents == 1 else 'cents'
    return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
  elif dollars:
    dollar_unit = 'dollar' if dollars == 1 else 'dollars'
    return '%s %s' % (dollars, dollar_unit)
  elif cents:
    cent_unit = 'cent' if cents == 1 else 'cents'
    return '%s %s' % (cents, cent_unit)
  else:
    return 'zero dollars'

def _expand_ordinal(text):
  return _inflect.number_to_words(text.group(0))

def _expand_number(text):
  num = int(text.group(0))
  if num > 1000 and num < 3000:
    if num == 2000:
      return 'two thousand'
    elif num > 2000 and num < 2010:
      return 'two thousand ' + _inflect.number_to_words(num % 100)
    elif num % 100 == 0:
      return _inflect.number_to_words(num // 100) + ' hundred'
    else:
      return _inflect.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
  else:
    return _inflect.number_to_words(num, andword='')

def _normalize_numbers(text):
  text = re.sub(_pounds_re, r'\1 pounds', text)
  text = re.sub(_dollars_re, _expand_dollars, text)
  text = re.sub(_decimal_number_re, _expand_decimal_point, text)
  text = re.sub(_ordinal_re, _expand_ordinal, text)
  text = re.sub(_number_re, _expand_number, text)
  return text



def _expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text


def clean_text(text, remove_comma=True):
  """
  clean text from symbols, remove (some) abbreviations, lowercase, and normaize numbers  
  
  Arguments:
  text -- text to clean
  remove_comma -- flag to remove commas
  Returns:
  text -- cleaned text
  """
  text = _lowercase(text)
  text = _normalize_numbers(text)
  text = _expand_abbreviations(text)
  text = _remove_symbols(text)
  if remove_comma:
    text = _remove_commas(text)
  return text


def _string2int(text, alphabet_size = 26, first_letter=96):
   text = [ord(letter)-first_letter for letter in text ]
   for i in range(len(text)):
    if text[i] == -50: # -50 = . dot
      text[i] = alphabet_size+1

    if text[i] == -52: # -50 = , comma
      text[i] = alphabet_size+2

   return text

def _padd(texts):
  return tf.keras.preprocessing.sequence.pad_sequences(texts, maxlen=None,
                                                       dtype='int32', padding='post', value=0.0)

def _one_hot_encode(texts, comma=False, alphabet_size = 26, first_letter=96):
  depth = alphabet_size+3 if comma else alphabet_size+2
  return tf.one_hot(texts, depth=depth, axis=2)
import os
import sys
import pprint as pp # debug purposes
import time # time how long each section takes
import multiprocessing as mp # To lookup info online faster
import argparse

# Lookup word information
from vocabulary.vocabulary import Vocabulary # online dictionary
import nltk # local dictionary
import wikipediaapi # for more advance definitions
import urllib # google search
from bs4 import BeautifulSoup # google search
import requests # google search
import webbrowser # google search

# Capture source image
import cv2 # for webcam usage
from Foundation import * # For osascript crap (applescript)

# Google Tesseract OCR
from PIL import Image, ImageEnhance # photo manipulation
import pytesseract # bindings
import numpy as np # more advance photo manipulation

# Google Vision OCR
from google.cloud import vision
from google.cloud.vision import types
import io

os.environ['NO_PROXY'] = '*' # https://bugs.python.org/issue30385

"""
I vaguely wondered about the HQ trivia game and automating to get an edge in
the game since there is nothing stopping you from googling for answers. Why
not automate it in effort to learn? Well here's my attempt...
"""

class HQTrivia():
    """
    A simple test to see if one can automate determining the best answer
    """

    def __init__(self):

        # This determines source location on where to caputer picture
        # QuickTime - MacOS has record feature for phone (best)
        # WebCam - Use OpenCV to capture photo (untested)
        self.use_quicktime = False
        self.use_webcam = False
        self.use_input = False

        # The filename of picture (no extension means we're capturing image)
        self.picture = 'source'

        # Default location of where to work on self.picture
        self.location = os.getcwd()

        # Replace with your own auth file name
        self.google_auth_json = 'blissend.json'

        # Default the language for wikipedia searches
        self.wiki = wikipediaapi.Wikipedia('en')
        self.vb = Vocabulary()

        # The OCR text
        self.raw = ''

        # The information we ultimately wanted to be analyzed
        self.question = ''
        self.answers = {}
        self.definitions = {}

        # For debugging
        self.verbose = False

    def debug(self, msg):
        print("hqtrivia-automation.py: " + str(msg))

    def capture(self, ftype='tiff'):
        """
        Simple function to select function to capture picture
        """

        if self.verbose:
            pre = "method - capture | "
            self.debug(pre + "choosing how to capture...")

        if self.use_input:
            if self.verbose:
                self.debug(pre + "input provided, don't capture")
            return

        # Set file type
        self.picture += '.' + ftype

        if self.use_quicktime:
            if self.verbose:
                self.debug(pre + "quicktime")
            self.quicktime(ftype)
        elif self.use_webcam:
            if self.verbose:
                self.debug(pre + "webcam")
            self.webcam()

    def quicktime(self, ftype='tiff'):
        """
        Takes screenshot of phone screen via AppleScript

        To use this open QuickTime player and do a movie recording. Select the
        drop down arrow next to record button and select your iPhone. This
        requires a wire connection to your computer using QuickTime. Remember,
        don't record anything. Having it show on screen is enough for a
        screencapture!

        1. Get window ID of QuickTime Player
        2. Tell script to run shell script command screencapture the window ID
        """

        if self.verbose:
            self.debug("method - quicktime | starting")
            start = time.time()

        full_path = os.path.join(self.location, self.picture)
        script = """tell application "QuickTime Player"
set winID to id of window 1
end tell
do shell script "screencapture -x -t tiff -l " & winID &"""
        script += ' " ' + full_path + '"'
        script = script.replace('tiff', ftype)

        s = NSAppleScript.alloc().initWithSource_(script)
        s.executeAndReturnError_(None)

        if self.verbose:
            diff = time.time() - start
            self.debug("method - quicktime | elapsed {!s}".format(diff))

    def webcam(self):
        """
        Takes screenshot using webcam.

        This is untested but here just in case it's needed. You need to figure
        out which camera to capture which unfortnately appears to be a discovery
        process of entering in numbers from 0 to higher until found. Also note,
        not all cameras have good controls and autofocus sucks for this.
        """

        if self.verbose:
            self.debug("method - webcam | starting")

        video = cv2.VideoCapture(1) # cam id (try from 0 and higher til found)
        video.set(cv2.CAP_PROP_AUTOFOCUS, 0) # turn the autofocus off
        video.set(3, 1920)
        video.set(4, 1080)
        cv2.namedWindow("HQ OCR Camera")
        #img_counter = 0
        while True:
            ret, frame = video.read()
            cv2.imshow("HQ OCR Camera", frame)
            if not ret:
                break
            k = cv2.waitKey(1)
            if k%256 == 27: # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32: # SPACE pressed
                img_name = self.picture # format with counter for multiple pics
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                #img_counter += 1
                break
        video.release()
        cv2.destroyAllWindows()

    def enhance(self):
        """
        Edit image readability for Google OCR that is suuuuuuuuuuuuuper...

        picky

        1. Eliminate background on buttons (answers)
        2. Turn to grayscale
        3. Make image BIG because Google OCR likes it big ;)
        4. Reduce the grayscale (eliminates button borders in good pictures)
        5. Make anything not white, black because google can't see color -_-
        """

        if self.verbose:
            self.debug("method - enhance | starting")
            start = time.time()

        # Replace buttons (answers) background color, incease size scale/DPI
        im = Image.open(self.picture)
        im = im.convert("RGBA")
        data = np.array(im)
        red, green, blue, alpha = data.T # Unpack the bands for readability
        gray_buttons = (red > 225) & (green > 225) & (blue > 225)
        data[..., :-1][gray_buttons.T] = (255, 255, 255)
        im = Image.fromarray(data)
        width, height = im.size
        im.crop((0, 300, width, height-400)).save(self.picture)
        #im.resize((round(width*3), round(height*3))).save(
            #self.picture, dpi=(600,600))

        # Make grayscale
        im = Image.open(self.picture)
        im = im.convert('RGBA')
        im = im.convert('L').save(self.picture)
        #exit()

        # Reduce the grayscale
        #im = Image.open(self.picture)
        #im = im.convert('RGBA')
        #data = np.array(im)
        #red, green, blue, alpha = data.T # Unpack the bands for readability
        #gray_triming = (red > 158) & (green > 158) & (blue > 158)
        #data[..., :-1][gray_triming.T] = (255, 255, 255)
        #Image.fromarray(data).save(self.picture)
        #exit()

        # Replace non white with black
        im = Image.open(self.picture)
        im = im.convert('RGBA')
        data = np.array(im)
        red, green, blue, alpha = data.T # Unpack the bands for readability
        non_white = (red < 255) & (green < 255) & (blue < 255)
        data[..., :-1][non_white.T] = (0, 0, 0)
        im = Image.fromarray(data)
        width, height = im.size
        im.resize((round(width*3), round(height*3))).save(self.picture)
        #exit()

        if self.verbose:
            diff = time.time() - start
            self.debug("method - enhance | elapsed {!s}".format(diff))

    def vision_ocr(self):
        """
        Google Cloud Vision

        The better OCR tool out there but requires additional setup. It is
        free under limitations.
        """

        # Credentials
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(
            self.location, self.google_auth_json)

        # Instantiates a client
        client = vision.ImageAnnotatorClient() # spits out shit, don't know why

        # The image file
        if not os.path.isfile(self.picture):
            full_path = os.path.join(self.location, self.picture)
        else:
            full_path = self.picture

        # Loads the image into memory
        with io.open(full_path, 'rb') as image_file:
            content = image_file.read()

        image = types.Image(content=content)

        # Performs text detection on the image file
        response = client.text_detection(image=image)
        text = response.text_annotations

        for t in text:
            self.raw = t.description
            break

        # Clean up text
        self.raw = self.raw.split('\n')
        self.debug("method - vision | raw - " + str(self.raw))
        self.raw.pop(0)
        self.raw.pop(0)
        self.raw.pop(0)
        if self.raw[0].lower() == "time's up":
            self.raw.pop(0)
        self.raw.pop(-1)

    def tesseract_ocr(self):
        """
        Google Tesseract OCR

        Finally read the image text if possible
        """

        # Include the below line, if you don't have tesseract in your PATH
        # Example tesseract_cmd: '/usr/local/bin/tesseract'
        #pytesseract.pytesseract.tesseract_cmd = '<fullpath_to_tesseract>'

        if self.verbose:
            self.debug("method - ocr | starting")
            start = time.time()

        # Get text from image (OCR)
        self.raw = pytesseract.image_to_string(
            Image.open(self.picture), config="-psm 11")

        # Clean it up
        self.raw = self.raw.split('\n')

        if self.verbose:
            self.debug("method - ocr | raw = " + str(self.raw))

        index = 0
        while index < len(self.raw):
            value = self.raw[index].lower()
            if len(value) < 1:
                self.raw.pop(index)
                #self.debug("method - ocr | delete [" + value + "]")
            else:
                index += 1

        if self.verbose:
            self.debug("method - ocr | raw - cleaned = " + str(self.raw))
            diff = time.time() - start
            self.debug("method - ocr | elapsed {!s}".format(diff))

    def parse(self):
        """
        Parser for the OCR text

        This is tricky because the OCR text won't always be the same. So
        adjustments may have to be tweaked here.
        """

        if self.verbose:
            self.debug("method - parse | starting")
            start = time.time()

        # Parse text into question and answer variables
        check_q = True
        counter = 1 # for counting answers
        for line in self.raw:
            #print(len(line), end=' ['+line+']\n')
            if check_q:
                if len(line) > 2:
                    if '?' not in line:
                        self.question += line + ' '
                    else:
                        self.question += line
                        check_q = False
            else:
                if 'Swipe left' not in line:
                    if len(line) > 0 and line != '-':
                        ans = line
                        self.answers[str(counter)] = {
                            "answer": ans,
                            "keywords": [],
                            "score": 0,
                            "index": str(counter)
                        }
                        self.definitions[ans] = []
                        counter += 1
                else:
                    break

        if self.verbose:
            self.debug("method - parse | question = " + str(self.question))
            self.debug("method - parse | answer = " + str(self.answers))
            diff = time.time() - start
            self.debug("method - parse | elapsed {!s}".format(diff))

    def lookup(self, index):
        """
        Gets information about answer to determine relevance to question

        This is a multiprocess function and therefore updated values have to be
        returned to parent process.
        """

        if self.verbose:
            pre = "method - lookup | "
            self.debug(pre + "starting")
            start = time.time()

        # Reference/copy of self values
        answers = self.answers[index]
        definitions = self.definitions[answers['answer']]
        value = answers['answer']
        question_nouns = ''
        for q in nltk.pos_tag(nltk.word_tokenize(self.question)):
            if q[1] == 'NN' or q[1] == 'NNP':
                question_nouns += " " + q[0]
        question_nouns = question_nouns.strip()

        # First get wikipedia information (the most helpful)
        time_wiki = time.time()
        try:
            page = self.wiki.page(value)
            if page.exists():
                definitions.append(page)
                definitions.append("[Wikipedia]: " + page.summary)
        except:
            self.debug(pre + "issue with wikipedia... ")
            self.debug(sys.exc_info()[0])
        if self.verbose:
            self.debug(pre + "wiki elapsed " + str(time.time() - time_wiki))

        # Google search
        time_gsearch = time.time()
        try:
            text = urllib.parse.quote_plus(value)
            url = 'https://google.com/search?q=' + text
            response = requests.get(url, timeout=2)
            soup = BeautifulSoup(response.text, 'lxml')
            results = ''
            for g in soup.find_all(class_='st'):
                results += " " + g.text
            definitions.append("[Google]: " + results.strip().replace('\n',''))
        except:
            self.debug(pre + "issue with google search... ")
            self.debug(sys.exc_info()[0])
        if self.verbose:
            self.debug(
                pre +
                "google search elapsed " +
                str(time.time() - time_gsearch))

        # Get dictionary definitions
        time_define = time.time()
        define = nltk.corpus.wordnet.synsets(value)
        if len(define) < 1:
            # Means local dictionary didn't find anything so search online
            try:
                define = self.vb.meaning(value, format='list')
                if define != False:
                    # There may be multiple difinitions so count them up
                    counter = 1
                    for d in define:
                        definitions.append(
                            "[Meaning " + str(counter) + "]: " + d)
                        counter += 1
            except:
                self.debug(pre + "issue with vocabulary... ")
                self.debug(sys.exc_info()[0])
        else:
            definitions.append("[Meaning]: " + define[0].definition())
        if self.verbose:
            self.debug(
                pre +
                "dictionary elapsed " +
                str(time.time() - time_define))

        # Get synonyms
        time_synonyms = time.time()
        no_definitions = True
        if type(define) == list:
            if len(define) > 0:
                no_definitions = False
                synonyms = [l.name() for s in define for l in s.lemmas()]
                # Remove duplicates
                s = []
                i = 0
                while i < len(synonyms):
                    if synonyms[i] in s:
                        synonyms.pop(i)
                    else:
                        s.append(synonyms[i])
                        i += 1
                definitions.append("[Synonyms]: " + ', '.join(s))
        if no_definitions:
            # Means local dictionary didn't find anything so search online
            try:
                synonyms = self.vb.synonym(value, format='list')
                if synonyms != False:
                    definitions.append("[Synonyms]: " + str(synonyms))
            except:
                self.debug("method - lookup | issue with vocabulary... ")
                self.debug(sys.exc_info()[0])
        if self.verbose:
            self.debug(
                pre +
                "synonyms elapsed " +
                str(time.time() - time_synonyms))

        # Score the answer
        if len(definitions) > 0:
            for define in definitions:
                if type(define) == str:
                    if "[Wikipedia]" not in define:
                        d = define.split(':') # remove pretag [Google]: bla blah
                        words = d[1].split(' ')
                else:
                    # This is for WIKIPEDIA sections which isn't a string
                    words = []
                    for i in page.sections:
                        words += i.text.split(' ')
                for w in words:
                    if len(w) > 2:
                        if w in question_nouns:
                            if w not in answers['keywords']:
                                answers['keywords'].append(w)
                                answers['score'] += 1

        if self.verbose:
            diff = time.time() - start
            self.debug("method - lookup | elapsed {!s} for {!s}".format(diff, index))

        # Send data back to parent process
        return answers, definitions, index

    def display(self):
        # Question
        print('\n\nQuestion - ' + self.question, end='\n\n')

        # Answers
        choice = {'index': [], 'score': 0}
        for a, ans in self.answers.items():
            if ans['score'] == choice['score']:
                choice['index'].append(a)
            if 'NOT' in self.question:
                if ans['score'] < choice['score']:
                    choice['index'] = [a]
                    choice['score'] = ans['score']
            else:
                if ans['score'] > choice['score']:
                    choice['index'] = [a]
                    choice['score'] = ans['score']
            print("Choice - " + ans['answer'] + ' - Score ' + str(ans['score']))
            for d in self.definitions[ans['answer']]:
                if type(d) == str:
                    if len(d) > 140:
                        print(d[0:140])
                    else:
                        print(d)
            print("[Keywords]: " + str(ans['keywords']))
            print("")

        # Choose answer
        if len(choice['index']) > 0:
            choose = []
            for i in choice['index']:
                choose.append(self.answers[i]['answer'])
            print("Answer - " + ', '.join(choose), end='')
            if 'NOT' in self.question:
                print(" - NOT keyword so lowest score is " +
                      str(choice['score']))
            else:
                print(" - highest score is " + str(choice['score']))
        else:
            print("Answer - Unknown")
        print("")

if __name__ == '__main__':
    start = time.time()

    # Setup command line options
    parser = argparse.ArgumentParser(
        description='Automate searching for answers in HQ Trivia')
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-q', '--quicktime',
        action='store_true', default=False,
        help="Use quicktime to capture source image"
    )
    group.add_argument(
        '-w', '--webcam',
        action='store_true', default=False,
        help="Use webcam to capture source image"
    )
    group.add_argument(
        '-i', '--input',
        action='store',
        help="Use image provided instead of capturing"
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true', default=False,
        help="Spit out debug information"
    )
    options = parser.parse_args()

    # Configure class with command option
    hq = HQTrivia()
    hq.verbose = options.verbose
    if options.quicktime:
        hq.use_quicktime = options.quicktime
        hq.use_webcam = False
        hq.use_input = False
    elif options.webcam:
        hq.use_quicktime = False
        hq.use_webcam = options.webcam
        hq.use_input = False
    elif len(options.input) > 0:
        hq.use_quicktime = False
        hq.use_webcam = False
        hq.use_input = True
        hq.picture = options.input
    if options.verbose:
        hq.verbose = options.verbose

    # Uncomment and use your own custom settings
    #hq.location = '/Full/path/to' # Defaults to script location (leave alone)
    #hq.google_auth_json = '<your_json>.json' (if you're using google vision)

    # Capture image first
    hq.capture()

    # Read the picture (use either Tesseract or Vision but not BOTH!!!)
    hq.vision_ocr() # Google Vision API
    #hq.enhance() # Google Tesseract
    #hq.tesseract_ocr() # Google Tesseract

    # Parse the picture text
    hq.parse()

    # Get information about answers (time consumping so do multiprocessing)
    with mp.Pool(3) as p:
        result = p.starmap_async(hq.lookup, ('1', '2', '3',)).get() # .get(timeout=5)
    for new in result:
        index = new[2]
        hq.answers[index] = new[0]
        hq.definitions[new[0]['answer']] = new[1]

    # Display the results!
    hq.display()

    diff = time.time() - start
    print("Total Time - {!s}".format(diff))
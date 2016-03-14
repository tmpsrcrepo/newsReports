import re,string
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np

class preprocessingFunc(object):
    @staticmethod
    #removeObj: 1=remove punctuation,2 remove every nonletter,
    #ConvertNum: True: converted it to 'Num'*len(digits), False: don't convert
    #split_
    def cleanText(text,removeObj = 1,ConvertNum=True,remove_stopwords=False,split_=True):
        if removeObj == 1:
            regex = re.compile('[%s]' % re.escape(string.punctuation))
            text = regex.sub('',text)
        elif removeObj == 2:
            text = text.sub('[^a-zA-Z]',' ',text)

        if not split_:
            return text
        
        
        words = text.lower().split()
        
        if remove_stopwords:
            stop = set(stopwords.words('english'))
            words = [w for w in words if not w in stops]

        if ConvertNum:
            
            words = ['NUM'*len(w) if w.isdigit() else w for w in words]
            

        return words

    @staticmethod
    def text_to_sentences(text,tokenizer,remove_stopwords=False):
        raw_sentences = tokenizer.tokenize(text.decode('utf8').strip())
        print raw_sentences
        sentences = [preprocessingFunc.cleanText(sentence) for sentence in raw_sentences if len(sentence)>0]
        sentences = [w for w in sentences if w]
        return sentences



#sentence = ' BRATTLEBORO &GT;&GT; The Brattleboro Police Department is investigating a sexual assault that is alleged to have happened on Sept. 23. According to a press release, at 8:30 p.m. that night officers responded to Brattleboro Memorial Hospital after receiving a report of the assault. The alleged victim told an officer the event occurred in the early morning hours of Sept. 23 in a wooded area near a local grocery store on Canal Street. Detectives from the Brattleboro Police Departments Criminal Investigation Division were called in to investigate the case. The scene was processed and evidence was gathered. As a result, on Sept. 24, Levy N. Pierce, 28, of Brattleboro, was arrested without incident on Western Avenue. Pierce is being held pending arraignment in Windham Superior Court, Criminal Division, for the offenses of aggravated sexual assault, aggravated assault, unlawful restraint in the second degree, and lewd and lascivious conduct. The investigation is ongoing. A man who told police he had set fire to the corporate offices of Entergy Vermont Yankee on Old Ferry Road in 2011 has been formerly cited with arson, according to a press release from the Brattleboro Police Department. On Sept. 20, 2011, at just past 3 a.m., the Brattleboro Fire Department responded to an interior box alarm activation at Vermont Yankee Joint Information Center on Old Ferry Road. TheJoint Information Center is a three-level building that contained offices and storage equipment for Entergy. It was determined a fire had occurred on the middle floor that was being utilized for media relations. The fire was concentrated to an office area, which sustained water and fire damage. A preliminary investigation was conducted by members of the Brattleboro Police and Fire Departments, as well as the Division of Fire Safety. Based on the preliminary findings the fire was determined to be arson. Additional resources were brought in to assist in the investigation to include but not limited to the Vermont State Police Fire Investigation Unit, FBI Joint Terrorism Task Force, and the Fire and Explosion Investigation Section of the Massachusetts State Police. Evidence was collected at the scene and surveillance video was reviewed from various locations, but all leads were exhausted before the case was categorized as a cold case, which remained active. On Sept 8, 2015, a detective from the Brattleboro Police Department\'s Criminal Investigation received information of a person of interest who said he had information related to the fire and a day later, Detective Lt. Michael Carrier and BFD Fire Investigator Lenny Howard traveled to Burlington to interview the person. At the completion of the interview it was determined that there was probable cause to cite Anthony M. Gotavaskas, age 32, of Montpelier with arson. Gotavaskas will appear in Windham District Court, Criminal Division, to answer to the charge at a later date. The investigation is ongoing and additional charges may be filed against Gotavaskas. Sept. 21 Following an investigation of a multi-vehicle crash on Putney Road, the Brattleboro Police Department determined John E. Quay, 74, of Putney, was driving a 1992 Toyota pickup north on Putney Road near the intersection of Chickering Drive when Jonah L. Koch, 17, or Guilford, who was stopped at the Chickering Drive stop sign, pulled his 1997 Chevrolet Camaro into traffic on Putney Road, colliding with Quay\'s vehicle. Koch was issued a ticket for failing to yield entering traffic. Both vehicles pulled over to the east side of the road after the collision, first Koch and then Quay about 100 yards ahead of him. Quay then put his vehicle in reverse and drove backward on the wrong side of the road, driving south in reverse in the northbound lanes. At the same time, Steven J. Baraby, of Hinsdale, N.H., was driving his 2000 Honda motorcycle south on Putney Road. Baraby attempted to turn east into a business located on Putney Road, and as he did this, Quay, who was still traveling backwards in the wrong direction and wrong lane, struck Barab\'s motorcycle.This collision caused minor damage to the truck and extensive damage to the motorcycle. Baraby was injured and transported to Brattleboro Memorial Hospital by emergency responders for treatment of non-life threatening injuries. The investigation into the first collision is complete. The investigation into the second collision is ongoing. The collision was caused by Quay and appropriate enforcement action will be taken at the conclusion of the investigation. Sept. 24 The Brattleboro Police Department arrested Lamar James, 25, of Brattleboro, and cited him with aggravated domestic assault. At 9:20 a.m. Police responded to a report of a citizen dispute in progress. Subsequent investigation led to the arrest of James. Officers of the Brattleboro Police Department responded to a residence on South Main Street for a report of a violent domestic dispute. Subsequent investigation led to the arrest of Mark J. Caslin, 26, of Brattleboro, for domestic assault. All persons named in the Police Log are innocent until proven guilty in a court of law. '
#from nltk.tokenize import TreebankWordTokenizer
#print preprocessingFunc.text_to_sentences(sentence,tokenizer=TreebankWordTokenizer())

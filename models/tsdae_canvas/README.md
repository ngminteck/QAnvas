---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:4747
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: 'Master of Technology in Enterprise Business Analytics page 10
    of 28 MTech EBAC Grad Cert Exam Sem I I 2023/2024 : Customer Analytics Question
    4 (Total: 8 marks) Additionally, the company wants to determine whether one of
    its new products – a green smartphone would be commercially successful. In carrying
    out the investigation, the company is interested in identifying those consumers
    who would purchase the new product vs those who would not. To assist in identifying
    potential purchasers, the company conducted a survey and devised rating scales
    on three characteristics (see A, B, C listed below) to see which ones are useful
    in differentiating likely purchasers vs non- purchasers. Linear Discriminant Analysis
    (LDA) was conducted on the survey results to analyze the purchasers vs non -purchasers
    . A : Repairability (ie Ease of Repair and Maintenance) B : Modular Upgrade &
    adaptability; C : Durability Question Marks 1 / 8 2 / 13 3 / 3 4 / 8 5 / 8 6 /
    10 TOTAL / 50 Cluster1 Cluster2 Cluster3 Cluster4 Her-Interstellar His-Interstellar
    Customer ID X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X[REDACTED_PHONE] retail
    outlet Singsong Interstellar X mobile phone. developed to interact with customers
    was successful. This application would engage with potential customers and hopefully
    persuade the customer to purchase the product. This application has been installed
    on both Singsong owned retail outlets and other independent retail outfits. outlet
    either purchase or not purchase the phone. Some characteristics of the possible
    purchase interactions were captured. The potential customers gender (1=Male, 2
    =Female) Whether they were accompanied or unaccompanied when they entered the
    shop 1=Customer was alone, 2 =Customer was accompanied The type of retail outlet
    = 1 if it is a Singsong owned retail outlet, = 2 if it is some other retail outlet
    (e.g. Best Donki, Kurts…) Whether the retail outlet employed the new interactive
    multimedia applications to interact with the customer in his assessment of the
    Singsong Interstellar X mobile phone 1=No 2=Yes Whether the retail outlet deployed
    sales assistants to interact with the customer in his assessment of the Singsong
    Interstellar X mobile phone 1=No 2=Yes The length of stay of the customer in the
    retail outlet (minutes) Whether the customer purchased the Singsong Interstellar
    X mobile phone ID Gender Acc retail type Multi Sales Asst Time Purchase [REDACTED_PHONE]'
  sentences:
  - '© [REDACTED_PHONE] NUS. All rights reserved. Page 7 XML as Standard Exchange
    Format •Popular in industry and text -processing community •With XML, we can insert
    tags onto a text to identify its parts. –Eg. , , , , etc. –Such tags are very
    useful as they allow selection/extraction of the parts to generate features for
    subsequent mining. •Many word processors allow documents to be saved as XML format
    Text Analytics Module 3: Get the Text Data Ready for Analysis Dr. Fan Zhenzhen
    NUS-ISS National University of Singapore Email: [REDACTED_EMAIL] © [REDACTED_PHONE]
    NUS. The contents contained in this document may not be reproduced in any form
    or by any means, without the written permission of ISS, NUS, other than for the
    purpose for which it has been supplied. ATA/S-TA/Text Preparation/v5.2 © [REDACTED_PHONE]NUS.
    All rights reserved. Page 1 Outline Overview of task Vectorization - Term-Document
    Matrix Vectorization - Embeddings © [REDACTED_PHONE] NUS. All rights reserved.
    Page 2 What is usually needed… Numeric Vectors Documents(Corpus) Lost glamor Rated
    2 by hotogama on Feb 23, High tea at Raffles! 2013 Rated 5 by PY on Feb 26, 2013
    Not what it was, but still a place to go While I d idtao n d.o t s ta y a t t
    h is h o t e l, I d id W e h a f te rn o o n te a in t h e t iff in go to r o
    h o a m veR a aa tt eR loda oA f4k f m l , e ba s ayn z wmdin h Ia i g l wde las
    ain sed ryn S v o i ion ct ng ea2a t 8p a o lFl r eeb ruary impre a s n s d e
    i d2t 0w w1a it3s h R s th u ae p t e ein r d . y G 5o r ub e ry a f t at s rca
    e ev r v e ic l- e g i a n n i d o n Feb 26, comm lo e v r e c l i y a f li o
    s o m d 2! o 0I f t ''1t s h3 e n o p t l a c c h e e . a I p l a b t u e t r
    it''s well v1 v2 V3 v4 v5 v6 v7 v8 v9 … went w b o a r c th k M w thyi e t hf
    ip rmsritc y ie mh ( upa srpeb psa rsn oiod xn t £ ow7 h 0aa sf v o oe rn tew
    oo)f Doc[REDACTED_PHONE] a fam fo o r u t s h de Si si a nam gpapbG poie o rinne
    retcam e S t eol l o inn f ctg t.ha ,T et ahi o n pen d l aL c w woene it g ah
    n Bada ltrih tutelse e bd it of sat d w ow ho n l etao t e bt x hep e eo h l rno
    iise nt tnhg oce r b e y ga ., r r Ao t a h um n e nd ud s s a t fta lf ot if
    e fo f r y r m o 5au a n k dae re t his hotel Doc[REDACTED_PHONE] minu i t n e
    s S , I l npe gaft at.r poT onh tri h e esr o e, e ued i s g sbph ya e mc d ia
    eal cnly iyd i eef dxy l-o ypu a atsr ed rain king Doc[REDACTED_PHONE] touris
    te t a ri p lo oinv f f et ahr! tem cooslopnhiearle a ttom tohsep thheer e of
    the bar Doc[REDACTED_PHONE] Have a drink in the Long Bar, place nowa,n wdi twhi
    tahwofuutl smuucshi ca asnwda rfumll of tourists. of overweNigohwt Ait tm h is
    re o oriw nc a t y hno esu. fWr ir s net u fwl t o silo hl r e ( l s ls o o if
    n y o th u e floor, … never go bdaidcnk’!t tTkhnheoenwre ,g aiot’r set o ns ottihtl
    lte oh teTh ioeffrrii,ng inRaol oLmon gfo r the more authBeanrt itch bapetl atshctee
    cs ru iicnrhr yS a iinnngd ta hfpaeom rwoeu.o srl du.s eAdb out Too bad ftohra
    Rt ayof£ful4e a0s.rae hderiandki nfogr i nfo!)o adn db uint tthhee choice eveninisg
    bpraitlrlioannist eadn adl mwohsetn e nmtiyre wly ifbey noisy mtouernisttios naell
    dta ikt iwnga sp hhoetor gbriarpthhdsa. y at How things change! the end of the
    meal a cake was presented, what amazing service. Suitable for tasks considering
    a document as an object, Stayed February 2013 e.g., document classification, document
    clustering © [REDACTED_PHONE] NUS. All rights reserved. Page 3 Lost glamor Rated
    2 by hotogama on Feb 23, High tea at Raffles! 2013 Rated 5 by PY on Feb 26, 2013
    Not what it was, but still a plac While I d idtao n d.o t s ta y a t t h is h
    o t e l, I d id W e h a f te rn o o n te a in t h e t iff in go to r o h o a m
    veR a aa tt eR loda oA f4k f m l , e ba s ayn z wmdin h Ia i g l wde las ain sed
    ryn S v o i ion ct ng ea2a t 8p a o lFl r ee impre a s n s d e i d2t 0w w1a it3s
    h R s th u ae p t e ein r d . y G 5o r ub e ry a f t at s rca e ev r v e ic l-
    e g i a n n i comm lo e v r e c l i y a f li o s o m d 2! o 0I f t ''1t s h3 e
    n o p t l a c c h e e . a I p l a b t u e t r it''s w went w b o a r c th k M
    w thyi e t hf ip rmsritc y ie mh ( upa srpeb psa rsn oiod xn t £ ow7 h 0aa sf
    v o oe rn tew oo a fam fo o r u t s h de Si si a nam gpapbG poie o rinne retcam
    e S t eol l o inn f ctg t.ha ,T et ahi o n pen d l aL c w woene it g ah n Bada
    ltr sat d w ow ho n l etao t e bt x hep e eo h l rno iise nt tnhg oce r b e y
    ga ., r r Ao t a h um n e nd ud s s a t fta lf ot if e fo f r y r m o 5au a n
    k dae r minu i t n e s S , I l npe gaft at.r poT onh tri h e esr o e, e ued i
    s g sbph ya e mc d ia eal cnly iyd i eef dxy l-o ypu a atsr ed touris te t a ri
    p lo oinv f f et ahr! tem cooslopnhiearle a ttom tohsep thheer e of t Have a drink
    in the Lo place nowa,n wdi twhi tahwofuutl smuucshi ca asnwda rfumll of t of overweNigohwt
    Ait tm h is re o oriw nc a t y hno esu. fWr ir s net u fwl t o silo hl r e ( l
    s ls o o if never go bdaidcnk’!t tTkhnheoenwre ,g aiot’r set o ns ottihtl lte
    oh teTh ioeffrrii,ng inRa more authBeanrt itch bapetl atshctee cs ru iicnrhr yS
    a iinnngd ta hfpaeom rwoeu.o srl du Too bad ftohra Rt ayof£ful4e a0s.rae hderiandki
    nfogr i nfo!)o adn db ui ted 2 by hotogama on Feb 23, High tea at Raffles! 13
    Rated 5 by PY on Feb 26, 2013 Not what it was, but still a plac ile I d idtao
    n d.o t s ta y a t t h is h o t e l, I d id W e h a f te rn o o n te a in t h
    e t iff in to r o h o a m veR a aa tt eR loda oA f4k f m l , e ba s ayn z wmdin
    h Ia i g l wde las ain sed ryn S v o i ion ct ng ea2a t 8p a o lFl r ee pre a
    s n s d e i d2t 0w w1a it3s h R s th u ae p t e ein r d . y G 5o r ub e ry a f
    t at s rca e ev r v e ic l- e g i a n n i mm lo e v r e c l i y a f li o s o m
    d 2! o 0I f t ''1t s h3 e n o p t l a c c h e e . a I p l a b t u e t r it''s
    w nt w b o a r c th k M w thyi e t hf ip rmsritc y ie mh ( upa srpeb psa rsn oiod
    xn t £ ow7 h 0aa sf v o oe rn tew oo am fo o r u t s h de Si si a nam gpapbG poie
    o rinne retcam e S t eol l o inn f ctg t.ha ,T et ahi o n pen d l aL c w woene
    it g ah n Bada ltr d w ow ho n l etao t e bt x hep e eo h l rno iise nt tnhg oce
    r b e y ga ., r r Ao t a h um n e nd ud s s a t fta lf ot if e fo f r y r m o
    5au a n k dae r nu i t n e s S , I l npe gaft at.r poT onh tri h e esr o e, e
    ued i s g sbph ya e mc d ia eal cnly iyd i eef dxy l-o ypu a atsr ed ris te t
    a ri p lo oinv f f et ahr! tem cooslopnhiearle a ttom tohsep thheer e of t Have
    a drink in the Lo ce nowa,n wdi twhi tahwofuutl smuucshi ca asnwda rfumll of t
    overweNigohwt Ait tm h is re o oriw nc a t y hno esu. fWr ir s net u fwl t o silo
    hl r e ( l s ls o o if ver go bdaidcnk’!t tTkhnheoenwre ,g aiot’r set o ns ottihtl
    lte oh teTh ioeffrrii,ng inRa re authBeanrt itch bapetl atshctee cs ru iicnrhr
    yS a iinnngd ta hfpaeom rwoeu.o srl du o bad ftohra Rt ayof£ful4e a0s.rae hderiandki
    nfogr i nfo!)o adn db ui ted 5 by PY on Feb 26, 2013 Not what it was, but still
    a plac d idtao n d.o t s ta y a t t h is h o t e l, I d id h a f te rn o o n te
    a in t h e t iff in m veR a aa tt eR loda oA f4k f m l , e ba s ayn z wmdin h
    Ia i g l wde las ain sed ryn S v o i ion ct ng ea2a t 8p a o lFl r ee d e i d2t
    0w w1a it3s h R s th u ae p t e ein r d . y G 5o r ub e ry a f t at s rca e ev
    r v e ic l- e g i a n n i r e c l i y a f li o s o m d 2! o 0I f t ''1t s h3 e
    n o p t l a c c h e e . a I p l a b t u e t r it''s w r c th k M w thyi e t hf
    ip rmsritc y ie mh ( upa srpeb psa rsn oiod xn t £ ow7 h 0aa sf v o oe rn tew
    oo t s h de Si si a nam gpapbG poie o rinne retcam e S t eol l o inn f ctg t.ha
    ,T et ahi o n pen d l aL c w woene it g ah n Bada ltr o n l etao t e bt x hep
    e eo h l rno iise nt tnhg oce r b e y ga ., r r Ao t a h um n e nd ud s s a t
    fta lf ot if e fo f r y r m o 5au a n k dae r S , I l npe gaft at.r poT onh tri
    h e esr o e, e ued i s g sbph ya e mc d ia eal cnly iyd i eef dxy l-o ypu a atsr
    ed i p lo oinv f f et ahr! tem cooslopnhiearle a ttom tohsep thheer e of t Have
    a drink in the Lo owa,n wdi twhi tahwofuutl smuucshi ca asnwda rfumll of t weNigohwt
    Ait tm h is re o oriw nc a t y hno esu. fWr ir s net u fwl t o silo hl r e ( l
    s ls o o if o bdaidcnk’!t tTkhnheoenwre ,g aiot’r set o ns ottihtl lte oh teTh
    ioeffrrii,ng inRa uthBeanrt itch bapetl atshctee cs ru iicnrhr yS a iinnngd ta
    hfpaeom rwoeu.o srl du d ftohra Rt ayof£ful4e a0s.rae hderiandki nfogr i nfo!)o
    adn db ui e to go t stay at this hotel, I did afternoon tea in the tiffin eR oda
    oA f4k f m l , e ba s ayn z wmdin h Ia i g l wde las ain sed ryn S v o i ion ct
    ng ea2a t 8p a o lFl r ee 3s h R s th u ae p t e ein r d . y G 5o r ub e ry a
    f t at s rca e ev r v e ic l- e g i a n n i o m d 2! o 0I f t ''1t s h3 e n o
    p t l a c c h e e . a I p l a b t u e t r it''s w hf ip rmsritc y ie mh ( upa
    srpeb psa rsn oiod xn t £ ow7 h 0aa sf v o oe rn tew oo m gpapbG poie o rinne
    retcam e S t eol l o inn f ctg t.ha ,T et ahi o n pen d l aL c w woene it g ah
    n Bada ltr ep e eo h l rno iise nt tnhg oce r b e y ga ., r r Ao t a h um n e
    nd ud s s a t fta lf ot if e fo f r y r m o 5au a n k dae r poT onh tri h e esr
    o e, e ued i s g sbph ya e mc d ia eal cnly iyd i eef dxy l-o ypu a atsr ed !
    tem cooslopnhiearle a ttom tohsep thheer e of t Have a drink in the Lo twhi tahwofuutl
    smuucshi ca asnwda rfumll of t Ait tm h is re o oriw nc a t y hno esu. fWr ir
    s net u fwl t o silo hl r e ( l s ls o o if k’!t tTkhnheoenwre ,g aiot’r set o
    ns ottihtl lte oh teTh ioeffrrii,ng inRa itch bapetl atshctee cs ru iicnrhr yS
    a iinnngd ta hfpaeom rwoeu.o srl du ayof£ful4e a0s.rae hderiandki nfogr i nfo!)o
    adn db ui eve nois How v1 v2 V3 v4 v5 v6 v7 v8 v9 … Doc[REDACTED_PHONE] Doc[REDACTED_PHONE]
    Doc[REDACTED_PHONE] Doc[REDACTED_PHONE] … Sources of Text Data Existing business
    Web processes Emails Customer Manuals call Social Legal Contracts feedback Web
    pages media documents (news, data Employee reports,etc. evaluations Transcripts
    Customer complaint Memos log Warranties Reports Etc. And many more… © [REDACTED_PHONE]
    NUS. All rights reserved. Page 4 File Preprocessing PDF TXT HTML XML JSON XML
    TXT Word Etc. ─Delete formatting tags EXCEL ─Remove special characters ─Detect
    and label different zones ─Determine sentence/paragraph boundaries Most TA tools
    provide functionality of importing text from some common formats. © [REDACTED_PHONE]
    NUS. All rights reserved. Page 5 In a pure text file… Amazing service Rated 5
    by travel-gini on Feb 26, 2021 Great location with a little bit of history, the
    staff make this hotel though Have a drink in the Long Bar, throw your nutshells
    on the floor, then go to the Tiffin Room for the best curry in the world. About
    £40a head for food but the choice is brilliant and when my wife mentioned it was
    her birthday at the end of the meal a cake was presented, what amazing service.
    Stayed February 2021 … © [REDACTED_PHONE] NUS. All rights reserved. Page 6 XML
    as Standard Exchange Format • Popular in industry and text-processing community
    DOC SUBJECT TOPIC TEXT • With XML, we can insert tags onto a text to identify
    its parts. – Eg. , , , , etc. – Such tags are very useful as they allow selection/extraction
    of the parts to generate features for subsequent mining. • Many word processors
    allow documents to be saved as XML format © [REDACTED_PHONE] NUS. All rights reserved.
    Page 7 What would an XML doc look like? Amazing service 5 26/02/2021 travel-gini
    Great location with a little bit of history, the staff make this hotel though
    Have a drink in the Long Bar, throw your nutshells on the floor, then go to the
    Tiffin Room for the best curry in the world. About £40a head for food but the
    choice is brilliant and when my wife mentioned it was her birthday at the end
    of the meal a cake was presented, what amazing service. Stayed February 2021 …
    © [REDACTED_PHONE] NUS. All rights reserved. Page 8 JSON Format • JavaScript Object
    Notation (JSON) is a standard text-based format for • representing structured
    data based on JavaScript object syntax Uses human-readable text to store and transmit
    data objects consisting of • attribute-value pairs and arrays Common data format
    with diverse uses in electronic data interchange, including that of web applications
    with servers © [REDACTED_PHONE] NUS. All rights reserved. Page 9 Our example in
    JSON format { “title”: “ “rating”: Amazing service”, “date”: “ 5, “by”: “ 26/02/2021”,
    “content”: “ travel-gini”, Great location with a little bit of history, the staff
    make this hotel though Have a drink in the Long Bar, throw your nutshells on the
    floor, then go to the Tiffin Room for the best curry in the world. About £40a
    head for food but the choice is brilliant and when my wife mentioned it was her
    birthday at the end of the meal a cake was presented, what amazing service. }
    Stayed February 2021” © [REDACTED_PHONE] NUS. All rights reserved. Page 10 Vectorization
    – Term Document Matrix © [REDACTED_PHONE] NUS. All rights reserved. Page 11 TDM
    – Pre-processing Term Document Matrix • (TDM) uses terms (keywords) as features
    for the vectors to represent the documents. • Common pre-processing steps (for
    English, may be different for other languages) – Tokenization – Case lowering
    – Stemming/lemmatization – Stopword removal – Punctuation removal © [REDACTED_PHONE]
    NU – S. Al N l rig u ht m s re e se r rv i e c d. removal Page 12 Tokenization
    • To break a stream of characters into smaller units called tokens, which can
    be words, subwords, or characters, depending on the methods • For TDM, word tokenization
    is used Great location with a little bit of history. – unigram model – every token
    is a single word. Great location with a little bit of history . – There are also
    bigram, trigram models, where each token is composed of two/three words. Great
    location location with with a a little little bit bit of of history history .
    © [REDACTED_PHONE] NUS. All rights reserved. Page 13 Tokenization Challenges space,
    tab, newline • Tokenization is done by identifying token delimiters ( ) ! ? “
    ” – Whitespace characters such as . , : - ‘ ’ – Punctuation characters like –
    Other characters etc. . , : • It seems simple, but… 12.34 12,345 12:34 – between
    numbers are part of the number . U.S.A. Dr. – can be part of an abbreviation or
    end of a sentence ’ – can be a closing internal quote, indicate a possessive,
    or be part of My friend’s isn’t another token © [REDACTED_PHONE] NUS. All rights
    reserved. Page 14 Vocabulary and Unknown Words unique • Vocabulary – the set of
    all tokens that a model can recognize (has encountered in a training corpus) –
    size: a large vocabulary size can capture more specific information, but increases
    memory usage and computational cost • Unknown words – Words not present in the
    vocabulary (but the model may encounter), also known as Out-of-Vocabulary words
    or OOV – Usually handled by replacing unknown words with a special token, like
    – Modern models deal with unknown words using subword decomposition. © [REDACTED_PHONE]
    NUS. All rights reserved. Page 15 Case Lowering case normalization • Also known
    as , to convert all tokens to lower case to reGmreoavte the vloacraiatitoionn
    of wowridths duea to casliett ldeifferbeitnceso.f history . great location with
    a little bit of history . • May cause issue in some contexts – E.g., “Apple” ->
    “apple”, “US” to “us” © [REDACTED_PHONE] NUS. All rights reserved. Page 16 Stemming/Lemmatization
    • A word may come in varied forms and therefore need to be converted into a standard
    form • Stemming can reduce the number of distinct features in a text corpus and
    increase the frequency of occurrence of some Inflectional Lemmatization individual
    features. – stemming (no change of POS) – Derivational • “apples” -> “apple”,
    “eating” -> “eat” – Stemming (with change of POS) • “production” -> “produce”
    – Other nUo.rSm.Aa.lisation (including case normalisation) USA US Refer to p5
    in Essential Linguistics © [REDACTED_PHONE] NUS. All rights reserved. Page 17
    Stemmers • Some well-known stemming algorithms for English Lovins Stemmer • single
    pass, longest-match • removing the longest suffix, ensuring the remaining by Julie
    Beth Lovins, stem is at least 3 characters long 1968 • reforming the stem through
    recoding transformations • Widely used Porter Stemmer • with implementations in
    various languages available online (C, java, Perl, python, C#, VB, Javascript,
    Tcl, by Martin Porter, 1980 Ruby, etc.) Snowball Stemmer • a framework for writing
    Stemming algorithms • Newer version of Porter Stemmer by Porter © [REDACTED_PHONE]
    NUS. All rights reserved. Page 18 How much stemming should be done? • An inflectional
    stemmer needs to be partly rule-based and partly dictionary-based. • Derivational
    stemming is more aggressive and therefore can reduce the number of features in
    a corpus drastically. However, meaning might be lost in the stemming process.
    Too aggressive stemming can result in loss of meaning and non-legitimate words
    without the support of a dictionary. © [REDACTED_PHONE] NUS. All rights reserved.
    Page 19 Stopword Removal • Some words are extremely common. They appear in almost
    all documents and carry little meaning. They are of limited use in text analytics
    applications. the, of, to, and, it – Functional words (conjunctions, prepositions,
    determiners, or pronouns) like , etc. Depending on the domain – A stopword list
    can be constructed to exclude them from analysis. – , other words may need to
    be included in the stopword list. great location with a little bit of history
    . © [REDACTED_PHONE] NUS. All rights reserved. Page 20 Stopword List filter dictionary
    exclusion dictionary • Also known as / • To support the stopword removal step
    in preprocessing preposition conjunction • A list of very common words – usually
    functional words like , , etc. – or words that are unimportant for the mining
    task a an because before • Example stopword list (not complete): about and been
    being above any before below after are being between again aren''t below both
    against as between but all at both by am be been … From http://www.ranks.nl/resources/stopwords.html
    © [REDACTED_PHONE] NUS. All rights reserved. Page 21 a mple stopw about above
    after again against all am an ord list (not and any are aren''t as at be because
    complete): been before being below between both been before being below between
    both but by … From http://www.ranks.nl/resources/stopwords.html Further clean-up
    • Punctuation removal • Number removal, etc. great location history . great location
    history © [REDACTED_PHONE] NUS. All rights reserved. Page 22 TDM - Indexing term-
    document matrix document-term matrix • Create the vector representation of documents
    ( or ) using “bag-of- words” approach T: term D: document w: weight of the term
    • Vector features: terms/keywords, usually only content words (adjectives, adverbs,
    nouns, and verbs). © [REDACTED_PHONE] NUS. All rights reserved. Page 23 Term Weighting
    • Binary – 0 or 1, simply indicating whether a word has occurred in the document
    (suitable for very short documents). term frequency • Frequency-based – , the
    frequency of words in the document, which provides additional information that
    can be used to contrast with other documents. amazing service lost glamour disappointbrilliant
    super expensive noisy … Doc[REDACTED_PHONE] Doc[REDACTED_PHONE] Doc[REDACTED_PHONE]
    Doc[REDACTED_PHONE] … © [REDACTED_PHONE] NUS. All rights reserved. Page 24 er
    docum amazing ents service . lost glamour disappoint brilliant super expensive
    noisy … Doc[REDACTED_PHONE] Doc[REDACTED_PHONE] Doc[REDACTED_PHONE] Doc[REDACTED_PHONE]
    … Frequent Word List • With frequency-based TDM, a list of words and their frequencies
    in the Global frequency corpus can be generated Document frequency – – how many
    times a word appears in the corpus – – how many unique documents contain the word
    • This list, sorted by frequency, can give us a rough idea of what the corpus
    is about. • Word Cloud is a nice visualization of such information. © [REDACTED_PHONE]
    NUS. All rights reserved. Page 25 Word Cloud: another example • Generated from
    http://worditout.com/word-cloud/make-a-new-one © [REDACTED_PHONE] NUS. All rights
    reserved. Page 26 Contrasting Two Groups • “What do you like least…” • “What do
    you like most…” © [REDACTED_PHONE] NUS. All rights reserved. Page 27 Other Weighting
    Methods • Normalized frequency – To deal with varied document length, since a
    long document definitely has more occurrences of terms than a short frequency
    of a term in a document normalized _ frequency = document total number of terms
    in the document tf-idf • inverse document frequency) – To modify the frequency
    of a word in a document by the perceived importance of the word(the , widely used
    in information retrieval • When a word appears in many documents, it’s considered
    unimportant. • When the word is relatively unique and appears in few documents,
    it’s important. © [REDACTED_PHONE] NUS. All rights reserved. Page 28 tf-idf Indexing
    tf-idf : • weighting tf-idf =tf *idf t,d t,d t – tf : term frequency – number
    of occurrences of term t in document d t,d – idf : inverted document frequency
    of term t t N idf = log t df t N : the total number of documents in the corpus
    df : the document frequency of term t, i.e., the number of documents that contain
    t the term. © [REDACTED_PHONE] NUS. All rights reserved. Page 29 tf-idf Indexing
    – An Example Note that in this example, stopwords and very common words are not
    removed, and terms are not reduced to root terms. http://www.miislita.com/term-vector/term-vector-3.html
    © [REDACTED_PHONE] NUS. All rights reserved. Page 30 Alternative Representation
    of TDM • The term document matrix is sparse, expected to have most of the values
    to be zero, since typically a document will only contain a small subset of the
    vocabulary in a corpus (ColumnIndex, Value) • It saves memory to store the matrix
    as a set of sparse vectors, where a row is represMenatteridx by a list of pairs,
    Sparse Vectors [REDACTED_PHONE], 5) (3, [REDACTED_PHONE], [REDACTED_PHONE], 3)
    (2, 1) (4, 6) © [REDACTED_PHONE] NUS. All rights reserved. Page [REDACTED_PHONE],
    5) (3, 2) (1, 4) (1, 3) (2, 1) (4, 6) Vectorization – Embeddings © [REDACTED_PHONE]
    NUS. All rights reserved. Page 32 Other Vectorization Methods • Word Embeddings:
    – A dense representation of words as vectors of real numbers in a continuous vector
    space with a much lower dimension – Learned by deep neural networks during a prediction
    task e.g. Word2Vec(2013), GloVe(2014), FastText(2016), etc. “Thou shall not make
    a machine in the likeliness of human mind…” - Look up embeddings - Calculate prediction
    © [REDACTED_PHONE] NUS. All rights reserved. Page 33 Word Embeddings Lookup •
    Using pre-trained word vectors, e.g., GloVe (Global Vectors for Word Representation)
    by Stanford I I • Lookup each token (word) inI the vectors. Lookup like like like
    shopping shopping shopping • Pros: the embeddings capture semantic or linguistic
    information of the words © [REDACTED_PHONE] NUS. All rights reserved. Page 34
    Contextualized Word Embeddings • Using a pre-trained (large) language model, e.g.,
    BERT, GPT2, etc. • Given text input, the model generates embedding for each word
    based on the context , better than lookup embeddingsI I input Pretrained output
    like Language like Model shopping shopping • Can handle different meanings of
    the same word with different context e.g., “Move the table closer to the window”
    vs. “Remove the first column in the table” • Great for tasks requiring word-level
    representation, like IE © [REDACTED_PHONE] NUS. All rights reserved. Page 35 Document
    Embeddings • Given each word in the document(any sequence of text) represented
    as an embedding vector, how to get the vector representation of the whole document?
    • Common approach: summarizing all the word vectors by averaging (or I summing)
    them averaging like I like shopping shopping doc2vec skip-thought, • However,
    there are also approaches to map documents to vectors, e.g., FastSent, Sentence-BERT,
    etc. using DL models to learn to represent documents( , ) © [REDACTED_PHONE] NUS.
    All rights reserved. Page 36 Summary Later… Vectorization (for document-level
    tasks) Parsing (syntactic and semantic) Named Entity Recognition TDM Embedding
    Shallow parsing/ chunking/partial parsing Part of Speech (POS) tagging File Preprocessing
    © [REDACTED_PHONE] NUS. All rights reserved. Page 37 Reference & Resources Speech
    & language processing • Jurafsky, Dan. . Pearson Education India, 2000. (continuously
    updated) Fundamentals of Predictive Text Mining • Weiss, Indurkhya, & Zhang. Chapter
    2 “From Textual Information to Numerical Vectors”, , Springer, 2010. • List of
    online word cloud generators – http://www.techlearning.com/default.aspx?tabid=67&entryid=364
    © [REDACTED_PHONE] NUS. All rights reserved. Page 38'
  - 'Master of Technology in Enterprise Business Analytics page 10 of 28 MTech EBAC
    Grad Cert Exam Sem I I 2023/2024 : Customer Analytics Question 4 (Total: 8 marks)
    Additionally, the company wants to determine whether one of its new products –
    a green smartphone would be commercially successful. In carrying out the investigation,
    the company is interested in identifying those consumers who would purchase the
    new product vs those who would not. To assist in identifying potential purchasers,
    the company conducted a survey and devised rating scales on three characteristics
    (see A, B, C listed below) to see which ones are useful in differentiating likely
    purchasers vs non- purchasers. Linear Discriminant Analysis (LDA) was conducted
    on the survey results to analyze the purchasers vs non -purchasers . A : Repairability
    (ie Ease of Repair and Maintenance) B : Modular Upgrade & adaptability; C : Durability
    Question Marks 1 / 8 2 / 13 3 / 3 4 / 8 5 / 8 6 / 10 TOTAL / 50 Cluster1 Cluster2
    Cluster3 Cluster4 Her-Interstellar His-Interstellar Customer ID X1 X2 X3 X4 X5
    X6 X7 X8 X9 X10 X11 X12 X13 X[REDACTED_PHONE] retail outlet Singsong Interstellar
    X mobile phone. developed to interact with customers was successful. This application
    would engage with potential customers and hopefully persuade the customer to purchase
    the product. This application has been installed on both Singsong owned retail
    outlets and other independent retail outfits. outlet either purchase or not purchase
    the phone. Some characteristics of the possible purchase interactions were captured.
    The potential customers gender (1=Male, 2 =Female) Whether they were accompanied
    or unaccompanied when they entered the shop 1=Customer was alone, 2 =Customer
    was accompanied The type of retail outlet = 1 if it is a Singsong owned retail
    outlet, = 2 if it is some other retail outlet (e.g. Best Donki, Kurts…) Whether
    the retail outlet employed the new interactive multimedia applications to interact
    with the customer in his assessment of the Singsong Interstellar X mobile phone
    1=No 2=Yes Whether the retail outlet deployed sales assistants to interact with
    the customer in his assessment of the Singsong Interstellar X mobile phone 1=No
    2=Yes The length of stay of the customer in the retail outlet (minutes) Whether
    the customer purchased the Singsong Interstellar X mobile phone ID Gender Acc
    retail type Multi Sales Asst Time Purchase [REDACTED_PHONE]'
  - 'Master of Technology in Artificial Intelligence Systems & Enterprise Business
    Analytics page 6 of 10 MTech EBAC/AIS Sample Paper: Practical Language Processing
    f. WeShare also decides to use voice print as one of the ways to access the system.
    There are two options (A and B) of voice print solu tions under consideration.
    The performances of them are shown in Figure 4, in which FAR (false acceptance
    rate) means the chance of an unauthorized user is accepted, and FRR (false reject
    rate) means the chance of an authorized user is rejected. If we would like to
    choose a system that has higher security level, which one should we choose? If
    we c oncern more about convenience of access to the system, which one should we
    choose? Justify your choices. Figure 4. Performance of Two Solutions (3 Marks)
    [Snippet 1] Meanwhile, the major European markets have all moved to the upside
    on the day. While the German DAX Index has risen by 0.4%, the French CAC 40 Index
    is up by 0.3% and the U.K.''s FTSE 100 Index is up by 0.2%. In commodities trading,
    crude oil futures are slip ping $0.16 to $64.89 a barrel after slumping $1.04
    to $65.05 a barrel on Monday. Meanwhile, after tu mbling $20.50 to $1,678 an ounce
    in the previous session, gold futures are spiking $29.90 to $1,707.90 an ounce.
    [Snippet 2] If we start thinking of the cryptocurrency as a cultural product,
    last week’s sudden jump in Dogecoin’s price makes sense. The boost came just after
    a meme-centric community managed to drive the share price of videogame retailer
    GameStop from US$20 to US$350 in mere days. One particularly interesting aspect
    of the Reddi t forum r/WallStreetBets – which coordinated the attack on the hedge
    fund that had effectively bet on GameStop’s share price falling – was how many
    users were having fun. Figure 3. Sample Text Snippets Target size Around 20,000
    articles Article selection method Randomly selected from the most recent articles,
    e.g. those uploaded in the past two months. Annotators interns (50 year‐1 college
    students on holiday) Labels ‘high’, ‘low’ Approach ‐ Brief the interns of the
    evaluation guidelines (summarized from the editor’s interview) ‐ Divide the dataset
    and assign each intern with 400 articles to be labeled as ‘high’ or ‘low’ t 1
    t 2 Figure 1. Language Model Review and critique the team’s data construction
    plan. Can the plan ensure a good dataset with consistent labels? If not, what
    are the problems in the plan? How would you improve it? Justify your answer using
    the information provided in the case study. Assuming that the ‘high’ articles
    used in the ‘Disruption Test’ are labelled correctly, the result of this test
    reveals the inadequacy of the existing transfer-learning approach adopted by the
    project team to capture the logical coherence of sentences. Identify two issues
    in the existing design of Language Model that are most likely the causes of this
    problem, and justify your answers. For each issue identified in b, propose how
    it can be resolved to improve the model’s ability to capture the logical coherence
    of an article, with limited GPU resources available. Describe the changes to the
    model’s network structure if any. Your answer should be based on the information
    given in the case study. While developing the ASR component for Audio Publishing,
    two candidate ASR systems (A and B) were evaluated to test their performances.
    The same set of testing speech recordings were passed to the two ASR systems,
    and two sample recognition results were recorded as shown in the following table.
    Please compare the result and explain the reasons for the difference between the
    two systems in terms of the major components of speech recognition systems. Sample
    1 Sample 2 Speech Content The consumer watchdog''s statement comes as tensions
    rise between private insurers and doctors over the use of Integrated Shield Plans
    panels. It takes about half an hour from Jurong East to Tiong Bahru. Result from
    system A The consumer watchdog''s statement comes as tensions rise between private
    insurers and doctors over the use of integrated shield plans panels. It takes
    about half an hour from rural East to Jonesboro Result from system B The consumer
    watchtowers detrimental comes as the tensions arise between private insurers at
    the doctors over the use of the integrated Shield plans panels It takes about
    half an hour from Jurong East to Tiong Bahru Speech Content The consumer watchdog''s
    statement comes as tensions rise between private insurers and doctors over the
    use of Integrated Shield Plans panels. It takes about half an hour from Jurong
    East to Tiong Bahru Result from system A The consumer watchdog''s statement comes
    as tensions rise between private insurers and doctors over the use of integrated
    shield plans panels. It takes about half an hour from rural East to Jonesboro
    Result from system B The consumer watchtowers detrimental comes as the tensions
    arise between private insurers at the doctors over the use of the integrated Shield
    plans panels It takes about half an hour from Jurong East to Tiong Bahru For auto
    speech content creation from text article, WeShare wants to try it in one of the
    most popular article categories among their users, ‘stock and market commentaries’.
    Two sample snippets of such articles are shown in Figure 3. What are the challenges
    for off-the-shelf TTS engines to work well for such ? What text normalization
    step you need to do if you expect the TTS system may not work well for some of
    the text content? Support your answer with examples from Figure 3. Suppose we
    decide to create a text normalization dataset with the help of the interns build
    a text normalization model with machine learning method. Please propose a method
    to achieve this goal. Kindly describe (1) the data set needs to be created, and
    (2) the machine learning method to be used. [Snippet 1] Meanwhile, the major European
    markets have all moved to the upside on the day. While the German DAX Index has
    risen by 0.4%, the French CAC 40 Index is up by 0.3% and the U.K.''s FTSE 100
    Index is up by 0.2%. In commodities trading, crude oil futures are slipping $0.16
    to $64.89 a barrel after slumping $1.04 to $65.05 a barrel on Monday. Meanwhile,
    after tumbling $20.50 to $1,678 an ounce in the previous session, gold futures
    are spiking $29.90 to $1,707.90 an ounce. [Snippet 2] If we start thinking of
    the cryptocurrency as a cultural product, last week’s sudden jump in Dogecoin’s
    price makes sense. The boost came just after a meme-centric community managed
    to drive the share price of videogame retailer GameStop from US$20 to US$350 in
    mere days. One particularly interesting aspect of the Reddit forum r/WallStreetBets
    – which coordinated the attack on the hedge fund that had effectively bet on GameStop’s
    share price falling – was how many users were having fun. WeShare also decides
    to use voice print as one of the ways to access the system. There are two options
    (A and B) of voice print solutions under consideration. The performances of them
    are shown in Figure 4, in which FAR (false acceptance rate) means the chance of
    an unauthorized user is accepted, and FRR (false reject rate) means the chance
    of an authorized user is rejected. If we would like to choose a system that has
    higher security level, which one should we choose? If we concern more about convenience
    of access to the system, which one should we choose? Justify your choices. Major
    Entity Action Time Amount Objective Entity Shell Egypt sell second half of 2021
    926 million Cheiron Petroleum Corporation Cair Energy Plc Major Entity Action
    Time Amount Objective Entity [REDACTED_PHONE]'
- source_sentence: 'Note: This page is strictly for Rough Work only and will not be
    marked.'
  sentences:
  - 'Note: This page is strictly for Rough Work only and will not be marked.'
  - © 2022 NUS. All rights reserved.Page 10 Today’s Agenda •Data Splits and Evaluation
    •Evaluation and Optimization –Over -/Underfitting –Regularization and Dropout
    •CNN for Text Classification –Convolutional Kernels for Text •Workshop •RNN and
    LSTM –RNN text Encoder –LSTM for Text Processing •Workshop a x1 b x2 c x3 … z
    x26 like 0 0 0 … 0 hate 1 0 0 … 0 good 0 0 0 … 0 enjoy 0 0 0 … 0 bad 0 1 0 … 0
    weight weight wb1 wp1 Wb2 wp2 Wb3 wp3 … … wb26 wp26 weight 0 wg[REDACTED_PHONE]
    R R R R R R R R R R R R R R R R R R R R [REDACTED_PHONE] Vanilla Not in used anymore
    LSTM Most popular Bi-directional GRU Simplified LSTM Faster but weaker* 1 0 1
    0 Dropout P =[REDACTED_PHONE] Dropout P =[REDACTED_PHONE]
  - Smarter applications with Edge-AI and SDR!
- source_sentence: '© 2019 NUS. All rights reserved.Page 45 Dialog Management (DM)
    •Decide “what to say” and “what to do” •There is no universally agreed definition
    •The complexity of DM depends on the specific tasks •Largely responsible for user
    satisfaction •DM Tasks: – Interaction Strategies – Error Handling and Confirmation
    Strategies – Dialogue State Tracking – Dialogue policy FoodType Local FoodType
    Indian FoodType Chinese FoodType ITALIAN PriceRange Cheap … … FoodType Local FoodType
    Indian FoodType Chinese FoodType ITALIAN PriceRange Cheap … … wordvec for word
    U S IsF IsR IsL IsP 1.2,2.4,5.9,0.1,[REDACTED_PHONE] FoodType Local FoodType Indian
    FoodType Chinese FoodType ITALIAN PriceRange Cheap … … would be i wordvec for
    word nfe U asi S ble IsF IsR IsL IsP 1.2,2.4,5.9,0.1,[REDACTED_PHONE] wordvec
    for context [REDACTED_PHONE]'
  sentences:
  - © 2025 National University of Singapore. All Rights Reserved History of Speech
    Recognition
  - 'ATA\ S-PRMLS \Day1b.ppt \V5.0 © 2024 National University of Singapore. All Rights
    Reserved 61 Workshop –Weka -MLP A BP network is auto- built according to the
    architecture setting Modify the network if necessary, such as adding/removing
    nodes and connections Click “Accept” to accept the network architecture Click
    “Start” to train the NN After training is done, click “Accept”, the NN model
    information and testing results is shown in “classifier output” Browse the results
    to find out model accuracy, confusion matrix, etc. Experiment with different
    parameters such as hidden layers, learning rate, epoch, etc. Compare the model
    performance Right click on the model item in the Result List and select “save
    model” to save the model PATTERN RECOGNITION AND MACHINE LEARNING SYSTEMS DAY
    1B Dr Zhu Fangming NUS-ISS [REDACTED_EMAIL] Not be reproduced in any form or by
    any means, without the written permission of ISS, NUS, other than for the purpose
    for which it has been supplied. ATA\S-PRMLS\Day1b.ppt\V5.0 © 2024 National University
    of Singapore. All Rights Reserved [REDACTED_PHONE] Neural Network Basics ATA\S-PRMLS\Day1b.ppt\V5.0
    © 2024 National University of Singapore. All Rights Reserved 2 Topics • Introduction
    to neural networks • Biological neuron and artificial neuron • General architecture
    of neural networks • Single-layer perceptron • Multilayer perceptron (MLP) • Backpropagation
    learning • Important issues in training ATA\S-PRMLS\Day1b.ppt\V5.0 © 2024 National
    University of Singapore. All Rights Reserved 3 Introduction to Neural Networks
    • In many real-world applications, computers are expected to perform complex pattern
    recognition tasks. • Complex pattern recognition involves thinking & learning
    • Learning involves both memorising and generalising • Recognition of complex
    patterns needs parallel processing • human can easily handle • conventional computing
    paradigms are not suitable to solve this type of problems • We therefore borrow
    features from physiology of brain as the basis for our new processing models —
    Artificial Neural Networks (ANN) or Neural Networks (NN) ATA\S-PRMLS\Day1b.ppt\V5.0
    © 2024 National University of Singapore. All Rights Reserved 4 Example: Character
    Recognition • Example: Character Recognition by Human Brain and NN 2 Neural 2
    Network (Black box) ATA\S-PRMLS\Day1b.ppt\V5.0 © 2024 National University of Singapore.
    All Rights Reserved 5 Some Definitions of Artificial Neural Networks • ... a neural
    network is a system composed of many simple processing elements operating in parallel
    whose function is determined by network structure, connection strengths, and the
    processing performed at computing elements or nodes. --------DARPA Neural Network
    Study (1988) • A neural network is a massively parallel distributed processor
    that has a natural propensity for storing experiential knowledge and making it
    available for use. It resembles the brain in two respects: • Knowledge is acquired
    by the network through a learning process. • Inter neuron connection strengths
    known as synaptic weights are used to store the knowledge. ----Haykin (1994) ATA\S-PRMLS\Day1b.ppt\V5.0
    © 2024 National University of Singapore. All Rights Reserved 6 From Biological
    Neuron to Artificial Neuron — Biological Inspiration for Artificial Neural Networks
    • The basic biological computing element — the neuron This extremely small computer
    is a multiple signal processor based on electrochemical processing principles
    Synapse Dendrites Cell Body Nucleus Axon ATA\S-PRMLS\Day1b.ppt\V5.0 © 2024 National
    University of Singapore. All Rights Reserved 7 From Biological Neuron to Artificial
    Neuron ... • A biological neuron is the basic biological computing element • A
    neuron is a small cell that receives electrochemical stimuli from multiple sources
    and responds by generating electrical impulses that are transmitted to other neurons
    or effector cells • There are something like 1010 to 1012 neurons in the human
    nervous system and each is capable of storing several bits of “information” •
    About 10% of the neurons are input and output, the remaining 90% are interconnected
    with other neurons which store information or perform various transformations
    on the signals being propagated through the network ATA\S-PRMLS\Day1b.ppt\V5.0
    © 2024 National University of Singapore. All Rights Reserved 8 From Biological
    Neuron to Artificial Neuron — An Artificial Neuron An artificial neuron — A Single
    Neural Computing Element • N input signals each weighted for its importance X1
    net = X1*W1 + X2*W2 +...+ Xn*Wn • Signals are added to produce a cumulative (net)
    input X2 W1 W2 OUTPUT f(net) INPUTS W3 • The neuron transforms the net input to
    produce an output X3 Wn o • net = Σxw o i i i o • Activation / transfer function
    Xn • hard-limiting function 1  0 if net 0 (false) • (0, 1) -> 1 (true) • (1,
    0) -> 1 (true) • (1, 1) -> 0 (false) • initial weights • w0 = 1, w1 = 1, w2 =
    1 • learning step α = 0.5 • Can we train a perceptron to solve this problem? Why?
    ATA\S-PRMLS\Day1b.ppt\V5.0 © 2024 National University of Singapore. All Rights
    Reserved 31 Single Perceptron and Linear Separability ... • In the activation
    function n Σ W X = θ i i i=1 forms a hyperplane in the n-dimensional space, dividing
    the space into two halves. Using θ as a threshold to produce output value (1 or
    0) means classifying the instances to two classes. When n = 2, the hyperplane
    becomes a line. • Linear separability • A data set is called “linearly separable”,
    when a linear hyper plane exists to place the instances of one class on one side
    and those of the other class on the other side. • A single perceptron can only
    solve a classification problem when it is linearly separable. • Many classification
    problems are not linearly separable. C • Sets A and B are linearly separable A
    A C C A B • Sets A and C, sets B and C are not linearly separable A C A B C B
    B B C ATA\S-PRMLS\Day1b.ppt\V5.0 © 2024 National University of Singapore. All
    Rights Reserved 32 Perceptron Learning Convergence Theorem If • given a set of
    input vectors, each of them with a desired output, and • each training case is
    presented to the network with positive probability Then • there is a procedure
    guaranteed to find a set of weights that give correct outputs if-and-only-if a
    set of weights exist for the task A single perceptron will converge only when
    the problem is linearly separable ATA\S-PRMLS\Day1b.ppt\V5.0 © 2024 National University
    of Singapore. All Rights Reserved 33 ADALINE & Its Learning Rule • ADALINE (ADAptive
    LINear Element) (Widrow, 1959) • a single unit similar to the perceptron • has
    a single output which receives multiple inputs, takes the weighted linear sum
    of the inputs and passes it to a bipolar function (which produces either +1 or
    -1, depending on the polarity of the sum) • MADALINE (Multiple ADALINE) • a network
    of ADALINEs • Learning rule of ADALINE Weighted Bipolar X1 • Widrow-Hoff rule
    (LMS, least mean square) Linear Sum W1 Function Σ W2 Y X2 Wn Wb Xn Bias ATA\S-PRMLS\Day1b.ppt\V5.0
    © 2024 National University of Singapore. All Rights Reserved 34 Widrow-Hoff Delta
    Rule • Linear Outputs: p p Σ Wik ith output for pattern p O = W X i ik k k • Target
    Output: T p Oi i Xk • Error Measure: OUTPUTS ½ Σ ( p p)2 ½ Σ ( p Σ p)2 o E(W)=
    T – O = T – w x o i i i ik k ip ip k INPUT o • Learning algorithm: • A form of
    gradient descent learning: change weight W proportional to the negative derivative
    ik of error. η is learning rate. ∂E p p p p p ∆w = –η = η(T – O )X =ηδ X ik ∂w
    i i k i k ik ATA\S-PRMLS\Day1b.ppt\V5.0 © 2024 National University of Singapore.
    All Rights Reserved 35 Multilayer Perceptron • Multilayer perceptron is a feedforward
    neural network with at least one hidden layer • It can form more complex decision
    regions to solve nonlinear classification problem • Each node in the first layer
    (above the input layer) can create a hyperplane. Each node in the second layer
    can combine hyperplanes to create convex decision regions. Each node in the third
    layer can combine convex regions to form concave regions. • The delta rule does
    not apply to training a multilayer network since the error of a hidden unit is
    not known. • The backpropagation learning method can overcome this difficulty
    X2 Y H1 H2 H3 X1 X1 X2 ATA\S-PRMLS\Day1b.ppt\V5.0 © 2024 National University of
    Singapore. All Rights Reserved 36 Multilayer Perceptron and Backpropagation Learning
    • Fully connected units • Feedforward signals only • One input-layer, one or more
    hidden-layers and one output layer • Nonlinear differentiable activation functions
    • Weights in hidden layers are adjusted to reduce aggregate errors in the output
    layer FORWARD SIGNAL FLOW • A two-stage process: • Propagate signals forward and
    then errors backward ERROR CORRECTION FLOW INPUT HIDDEN HIDDEN OUTPUT LAYER LAYER
    #1 LAYER #N LAYER ATA\S-PRMLS\Day1b.ppt\V5.0 © 2024 National University of Singapore.
    All Rights Reserved 37 Steps of Backpropagation Algorithm 1. Initialize the weights
    to small random numbers 2. Randomly select a training pattern pair (xp, tp) and
    present the input pattern xp to the network. Compute the corresponding network
    output pattern zp 3. Compute the error Ep for pattern (xp, tp) 4. Backpropagate
    the errors according to the BP weight adjustment formulas (given later) 1 P 5.
    Test the mean square error (MSE) over P training patterns: E = ∑ Ep P p=1 If the
    MSE is below the required threshold, stop. Otherwise, repeat steps 2-5. 6. Test
    for generalization performance if appropriate ATA\S-PRMLS\Day1b.ppt\V5.0 © 2024
    National University of Singapore. All Rights Reserved 38 BP Errors and Weight
    Updates • Total error over all training patterns p and m output units: p E = ∑
    Ep Total error, all training patterns tot p=1 1 m Error for pattern p Ep = ∑ (t
    − z )2 k k 2 k=1 z = f (I ) = f ( ∑ w y ) = f ( ∑ w f (H )) = k k j kj j j kj
    j = f ( ∑ w f ( ∑ v x )) j kj i ji i Vji Σ Σ Wkj f f z — computed output Tk k
    Ik Xi Zk Hj Yj f(Ik) Ek t — target output f(Hj) k Zk+1 Ek+1 o o Tk+1 o INPUT HIDDEN
    OUTPUT UNITS UNITS UNITS ATA\S-PRMLS\Day1b.ppt\V5.0 © 2024 National University
    of Singapore. All Rights Reserved 39 BP Errors and Weight Updates (cont.) • For
    output units k = 1, 2, ..., m, adjust the weights to reduce the error for each
    pattern p (Gradient descent) ∂Ep η m ∂(tp − zp)2 ∆w = −η = − ∑ k k kj ∂w 2 ∂w
    kj k=1 kj Dropping superscripts p ∂( )2 ∂( ( ))2 t − z t − f I ∂ I ( ) ′( ) k
    k = k k k = −2 t −z f I y ∂ w ∂ I ∂w k k k j kj k kj So ∂Ep η m ∂(tp − zp)2 ∆w
    = −η = − ∑ k k kj ∂w 2 ∂w kj k=1 kj = η(t − z ) f ′(I )y =ηδ y k k k j k j vji
    yj wkj zk xi Hj Ik • For hidden units, use the chain rule, get ∆v =ηf ′(H )x ∑δ
    w ji j i k kj k ATA\S-PRMLS\Day1b.ppt\V5.0 © 2024 National University of Singapore.
    All Rights Reserved 40 Summary of Backpropagation Algorithm • Weights initialization
    Set all weights to random numbers following Uniform distribution in the range
    (-1,1) • Calculation of activation • the activation level of an input unit is
    determined by the instance presented to the network • the activation level O of
    a hidden and output unit is determined by j Σ O = f( W O - θ) , where W is the
    weight from an input O, θ is the node threshold, f is the transfer j ji i j ji
    i j function. • Weight Updating 1) start at the output nodes and work backward
    to the hidden layers recursively, adjust weights by W (t+1) = W (t) + ∆W ji ji
    ji where W (t) is the weight from unit i to j at time t (or t-th iteration), ∆W
    is the weight adjustment ji ji ATA\S-PRMLS\Day1b.ppt\V5.0 © 2024 National University
    of Singapore. All Rights Reserved 41 Summary of Backpropagation Algorithm (cont.)
    (Assume a sigmoid function f(a) = 1/[1 + e-a] is used) • Weight Updating (cont.)
    2) The weight change is computed by ∆W = ηδO ji j i where η is a trial-independent
    learning rate (0 Anaconda Prompt: • conda create -n prmls python=3.7 • conda activate
    prmls • conda install numpy matplotlib jupyter pandas scikit-learn==0.21.2 keras==2.2.4
    pydot pydotplus • pip install neupy • Navigate to your working directory, eg.
    “d:\myfolder” • Run “jupyter notebook” • Now you can open .ipynb files within
    your browser ATA\S-PRMLS\Day1b.ppt\V5.0 © 2024 National University of Singapore.
    All Rights Reserved 63 Workshop- Python- Scikit-Learn & Keras Problem Description:
    Diabetes Prediction • This dataset is originally from the National Institute of
    Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to
    diagnostically predict whether or not a patient has diabetes, based on certain
    diagnostic measurements included in the dataset. • The datasets consists of several
    medical predictor variables and one target variable, Outcome. Predictor variables
    includes the number of pregnancies the patient has had, their BMI, insulin level,
    age, and so on. Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes,
    R.S. (1988).Using the ADAP learning algorithm to forecast the onset of diabetes
    mellitus.In Proceedings of the Symposium on Computer Applications and Medical
    Care(pp. 261--265). IEEE Computer Society Press. ATA\S-PRMLS\Day1b.ppt\V5.0 ©
    2024 National University of Singapore. All Rights Reserved 64 Workshop- Python-
    Scikit-Learn & Keras • Open the jupyter notebook provided for this workshop. •
    As you go through the notebook, make sure you understand how the NN models are
    built. (you can save notes as markdown in the notebook). • Check and compare the
    model performance. • Experiment with different parameter settings. • Save your
    notebook with the cell output and upload it to Canvas. Last ATA\S-PRMLS\Day1b.ppt\V5.0
    © 2024 National University of Singapore. All Rights Reserved 65'
  - '© 2019 NUS. All rights reserved.Page 45 Dialog Management (DM) •Decide “what
    to say” and “what to do” •There is no universally agreed definition •The complexity
    of DM depends on the specific tasks •Largely responsible for user satisfaction
    •DM Tasks: – Interaction Strategies – Error Handling and Confirmation Strategies
    – Dialogue State Tracking – Dialogue policy FoodType Local FoodType Indian FoodType
    Chinese FoodType ITALIAN PriceRange Cheap … … FoodType Local FoodType Indian FoodType
    Chinese FoodType ITALIAN PriceRange Cheap … … wordvec for word U S IsF IsR IsL
    IsP 1.2,2.4,5.9,0.1,[REDACTED_PHONE] FoodType Local FoodType Indian FoodType Chinese
    FoodType ITALIAN PriceRange Cheap … … would be i wordvec for word nfe U asi S
    ble IsF IsR IsL IsP 1.2,2.4,5.9,0.1,[REDACTED_PHONE] wordvec for context [REDACTED_PHONE]'
- source_sentence: 'Accelerating Digital Excellence Copyright National University
    of SingaporeLive or Near-live Translation • Real time translation: start while
    sentence is spoken Source: S. Zhang et al, Wait-info Policy: Balancing Source
    and Target at Information Level for Simultaneous Machine Translation, EMNLP 2022.
    Agenda • Day 3 • 1: Speech processing basics • 2: Speech recognition (Speech-to-text)
    • 3: Case studies: Integrating speech recognition and NLP solutions • Day 4 •
    4: Speech synthesis (Text-to-speech) • 5: Voice conversion and generation • 6:
    Spoken dialogue system (Spoken chatbot) 2 Copyright National University of Singapore
    Accelerating Digital Excellence ASR in Recent Years Source: D. Jurafskyand J.H.
    Martin, Speech and Language Processing. 46 Copyright National University of Singapore
    Accelerating Digital Excellence Evolution of AI systems AI systems perform ↑ better
    than humans ↓ AI systems perform worse Handwriting recognition Image recognition
    Language understanding Speech recognition Reading Reading comprehension comprehension
    Source: Kiela et al., Dynabench: Rethinking Benchmarking in NLP 48 Copyright National
    University of Singapore Accelerating Digital Excellence Evolution of ASR methods
    Source: Labellerr 49 Copyright National University of Singapore Accelerating Digital
    Excellence 75 A good tutorial: https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452
    Paradigm Shift • serves as a foundational layer upon which specialized applications
    are built. • involves extensive training data • being adapted to wide range of
    downstream tasks 78 Copyright National University of Singapore Accelerating Digital
    Excellence Whisper Model • Trained on 680k hours of multilingual and multitask
    supervised data • Three tasks: speech recognition, speech translation, and language
    recognition • #model parameters: 39M, 74M, 244M, 769M, 1550M 80 Source: https://openai.com/research/whisper
    Copyright National University of Singapore Accelerating Digital Excellence Wav2Vec2.0:
    Self Supervised Learning 81 Copyright National University of Singapore Accelerating
    Digital Excellence ChatGPT: LLM with Speech Capabilities 87 Copyright National
    University of Singapore Accelerating Digital Excellence Guidelines for Acoustic
    Modeling Units • Very phonetic languages (e.g., Spanish, German) • Letter-based
    or byte pair encoding (BPE) based acoustic modeling is effective. • Reasonably
    phonetic languages (e.g., English) • BPE-based acoustic model is preferred. •
    Letter-based acoustic modeling is also viable if large training datasets are available.
    • Non-phonetic or least phonetic languages (e.g., Chinese) • It is recommended
    to use a pronunciation dictionary to map words to modeling units. 90 Copyright
    National University of Singapore Accelerating Digital Excellence Further reading
    on Acoustic Modeling • Fine-Tune Whisper For Multilingual ASR with Transformers
    • https://huggingface.co/blog/fine-tune-whisper • Espnet: End-to-end speech processing
    • https://github.com/espnet/espnet • Kaldi: HMM-DNN acoustic modeling • https://kaldi-asr.org/
    • https://kaldi-asr.org/doc/tutorial.html • https://kaldi-asr.org/doc/kaldi_for_dummies.html
    • https://github.com/kaldi-asr/kaldi/tree/master/egs • HMM-based monophone and
    context-dependent triphone: • https://jonathan-hui.medium.com/speech-recognition-asr-model-training-
    90ed50d93615 91 Copyright National University of Singapore Accelerating Digital
    Excellence Further Reading on N-gram Language Models • Overview • https://web.eecs.umich.edu/~wangluxy/courses/eecs498_wn2021/slides_ee
    cs498_wn21/lm.pdf • LM training with SRILM • https://cmusphinx.github.io/wiki/tutoriallmadvanced/
    • LM linear interpolation/Building large n-gram LMs with SRILM • https://joshua.apache.org/6.0/large-lms.html
    • Morph n-gram model • http://research.spa.aalto.fi/speech/s895150/ex3.html 92
    Copyright National University of Singapore Accelerating Digital Excellence Open-source
    Python Speech Recognition • Installation: pip install SpeechRecognition • Library
    for performing speech recognition • Supported engines: • CMU Sphinx (works offline)
    • Google Speech Recognition • Google Cloud Speech API • Wit.ai • Microsoft Bing
    Voice Recognition • Houndify API • IBM Speech to Text • Snowboy Hotword Detection
    (works offline) • OpenAI Whisper (works offline) 107 Copyright National University
    of Singapore Accelerating Digital Excellence Open-source Python Speech Recognition
    • Recognize speech input from the microphone • Transcribe an audio file • Save
    audio data to an audio file • Calibrate the recognizer energy threshold for ambient
    noise levels • Listening to a microphone in the background • Website: https://pypi.org/project/SpeechRecognition/
    108 Copyright National University of Singapore Accelerating Digital Excellence
    Integrating speech recognition and NLP Possible applications: 1. Voice Search
    2. Customer Service Analysis 3. Speech Translation 4. Meeting Transcription and
    Summary Punctuation and capitalization Inverse text normalization restoration
    Copyright National University of Singapore Accelerating Digital Excellence Voice
    search • Find information using spoken queries, e.g. voice search in Taobao app
    • Text output from ASR often needs to transformed or normalized to match the input
    format expected by an existing text-based search system • e.g. iPhone sixteen
    sixty four g b => iPhone 16 64GB • Optimization for product terms (brand or product
    names) • Real me => Realme • Red me => Redmi Copyright National University of
    Singapore Accelerating Digital Excellence Quality Control of Customer Service
    • Customer and agent sentiment (from text and speech) • Speech sentiment analysis
    on voice characteristics such as pitch, loudness, etc • Text sentiment analysis
    • Non-talk time • Talk speed • Interruptions – Still challenging! Copyright National
    University of Singapore Accelerating Digital Excellence Speech Translation • Applications
    of machine translation include: • Cross-border communication • Localization of
    websites (and other digit content) • Language learning • Speech translation enables
    users to watch foreign videos, such as films and lectures, in their own language
    Source: https://www.mdpi.com/[REDACTED_PHONE]/13/15/8900 Copyright National University
    of Singapore Accelerating Digital Excellence Evolution of Machine Translation
    Methods Source: https://medium.com/free-code-camp/a-history-of-machine-translation-from-the-cold-war-to-deep-learning-f1d335ce8b5
    118 Copyright National University of Singapore Accelerating Digital Excellence
    Evolution of Machine Translation Methods 119 Copyright National University of
    Singapore Accelerating Digital Excellence Evaluation Metrics • Bilingual Evaluation
    Understudy (BLEU) Score • Calculated by comparing n-grams of machine-translated
    sentences to those of human-translated sentences. • Higher scores represent better
    MT performances. • BLEU scores may decreases as sentence lengths increase 120
    Copyright National University of Singapore Accelerating Digital Excellence Catastrophic
    Errors in MT • Generation of profanity • Eliminate words that appear in language-specific
    offensive word list. • BUT offensive language is not limited to specific words
    • Generation of violent or inciting content. • Reversal of intended meaning. •
    Mistranslation of proper names. 121 Copyright National University of Singapore
    Accelerating Digital Excellence Practical Challenges in Speech Translation • Misrecognized
    words lead to inaccurate or nonsensical translations. • Out-of-vocabulary words
    and domain-specific terms can be mistranslated or omitted. • MT systems for low-resource
    languages often exhibit poor accuracy, limiting their real-world utility. • Hesitations,
    false starts and filler words (“um”, “uh”) in ASR output lead to awkward or inaccurate
    MT output. • Inaccurate segmentation can confuse the MT model and reduce translation
    quality. • Live or near-live translation requires low latency. This can conflict
    with the need for high accuracy. 122 Copyright National University of Singapore
    Accelerating Digital Excellence Live or Near-live Translation • Real time translation:
    start while sentence is spoken I am going to talk today about energy and climate.
    Heute spreche ich zu Ihnen über Energie und Klima. • Subtitles: have to be readable
    in limited time • Dubbing: sync up with video of speaker’s mouth movement Copyright
    National University of Singapore Accelerating Digital Excellence Live or Near-live
    Translation • Real time translation: start while sentence is spoken Source: R.
    Zhang et al, Dynamic Sentence Boundary Detection for Simultaneous Translation,
    Proceedings of the 1st Workshop on Automatic Simultaneous Translation. Copyright
    National University of Singapore Accelerating Digital Excellence Live or Near-live
    Translation • Real time translation: start while sentence is spoken Source: S.
    Zhang et al, Wait-info Policy: Balancing Source and Target at Information Level
    for Simultaneous Machine Translation, EMNLP 2022. Copyright National University
    of Singapore Accelerating Digital Excellence Handling Context Across Multiple
    Sentences • Translation of pronouns may require co-reference 129 Copyright National
    University of Singapore Accelerating Digital Excellence Meeting Transcription
    Otter.ai Copyright National University of Singapore Accelerating Digital Excellence
    Meeting Summarization Evaluation metrics for summarization: BLEU, ROUGE Copyright
    National University of Singapore Accelerating Digital Excellence Practical Challenges
    in Meeting Summarization • Misrecognized words lead to inaccurate summary. • Business,
    medical, or legal meetings contain domain-specific terms that need accurate recognition
    and summarization. • Hesitations, false starts and filler words (“um”, “uh”) in
    ASR output lead to awkward or inaccurate summary. • Identifying and differentiating
    speakers correctly is crucial for attributing statements accurately. • Proper
    co-reference resolution of pronouns and vague references (this, that, those things)
    is important for accurate attribution of actions and decisions. • Different expectations
    of what is ”important” in a meeting. 132 Copyright National University of Singapore
    Accelerating Digital Excellence Infrastructure (GPU) Consideration for Production
    • Find something that fits your budget • Key performance metrics to be considered:
    • Token Per Second (TPS) = (Input Tokens + Output Tokens) / Total Turnaround Time
    • Time To First Token (TTFT) • GPU Usage: starting with T4 16GB GPU that works
    well for 3B or 8B models • Consider quantization and distillation of models (i.e.
    smaller models) • Consider small models with few-shot prompting. Source: https://rumn.medium.com/benchmarking-llm-performance-token-per-second-tps-time-to-first-token-ttft-and-gpu-usage-8c50ee8387fa
    https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html https://medium.com/aimonks/what-is-quantization-and-distillation-of-models-a67e3a2dc325
    138 Copyright National University of Singapore Accelerating Digital Excellence
    Gartner Hype Cycle 139 Copyright National University of Singapore Accelerating
    Digital Excellence'
  sentences:
  - 'ATA\ S-PSUPR \Day1b.ppt \V3.0 © 2024 National University of Singapore. All Rights
    Reserved 33 Bayesian Classification •It performs probabilistic prediction, i.e.,
    predicts class membership probabilities, based on Bayes’ Theorem. •Let Xbe a data
    sample: class label is unknown •Let H be a hypothesis that X belongs to class
    Ci •Classification is to determine P(H| X), (i.e., posteriori probability): the
    probability that the hypothesis holds given the observed data sample X •P(H) (
    prior probability ): the initial probability •P(X): probability that sample data
    is observed •P(X|H) ( likelihood ): the probability of observing the sample X,
    given that the hypothesis holds)()()|()|(XXXPHPH PHP= PROBLEM SOLVING USING PATTERN
    RECOGNITION DAY 1B Dr Zhu Fangming NUS-ISS National University of Singapore [REDACTED_EMAIL]
    Not be reproduced in any form or by any means, without the written permission
    of ISS, NUS, other than for the purpose for which it has been supplied. ATA\S-PSUPR\Day1b.ppt\V3.0
    © 2024 National University of Singapore. All Rights Reserved [REDACTED_PHONE]
    HOW TO ANALYSE, MODEL AND SOLVE PATTERN RECOGNITION PROBLEMS ATA\S-PSUPR\Day1b.ppt\V3.0
    © 2024 National University of Singapore. All Rights Reserved 2 Topics • Important
    steps in solving pattern recognition problems • Important issues for pattern recognition:
    data pre- processing, feature selection, model evaluation. ATA\S-PSUPR\Day1b.ppt\V3.0
    © 2024 National University of Singapore. All Rights Reserved 3 Pattern Recognition
    Process with Supervised Learning ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved 4 Models, Features and Classes • A pattern is
    represented by a set of d features, or attributes, viewed as a d-dimensional feature
    vector. , )T 𝟏𝟏 𝟐𝟐 𝑑𝑑 𝑿𝑿 = (𝒙𝒙 𝒙𝒙 , … 𝒙𝒙 x: input vector Classification / y: class
    label (pattern with Regression /regression features) Model p(x,y) How do we model
    p(x,y)? ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of Singapore. All
    Rights Reserved 5 Data Pre-processing  Data cleaning Fill in missing values,
    smooth noisy data, identify or remove outliers, and • resolve inconsistencies
     Data integration Integration of multiple databases •  Data transformation Normalization
    and aggregation •  Data reduction Dimensionality reduction - feature selection
    • Numerosity reduction – select/ sample records • ATA\S-PSUPR\Day1b.ppt\V3.0 ©
    2024 National University of Singapore. All Rights Reserved 6 Normalization • Normalization
    & feature scaling techniques are important for many machine learning algorithms.
    • Min-Max scaling [0,1] • Z-score (standardization) • Use the same parameters
    on the test dataset and new unseen data. ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National
    University of Singapore. All Rights Reserved 7 Feature Selection • Curse of dimensionality
    • Retain only "useful" (discriminatory) information and avoid overfitting. • Reasons
    to reduce the number of features: • Computational complexity • Generalization
    properties ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of Singapore.
    All Rights Reserved 8 Dimensionality Reduction • Principal Component Analyses
    (PCA) and Linear Discriminant Analysis (LDA) can be used. • Linear Discriminant
    Analysis (LDA) tries to identify attributes that account for the most variance
    between classes. • In particular, LDA, in contrast to PCA, is a supervised method,
    using known class labels. ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved 9 Data Partition and Preparation Training set
    vs. test set vs. validation set Cross-validation https://www.datasciencecentral.com
    ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of Singapore. All Rights
    Reserved 10 Cross Validation ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved 11 Learning from Imbalanced Data • Techniques
    for Learning from imbalanced data: • Data Augmentation • Custom Loss Function
    Z=0 Z=1 • Fraud detection • Churn Modeling • Anomaly detection ATA\S-PSUPR\Day1b.ppt\V3.0
    © 2024 National University of Singapore. All Rights Reserved 12 Model Evaluation
    • Error Measures • Overtraining/overfitting • Confusion Matrix • ROC Charts •
    Gains Chart/ Lift Chart ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved 13 Overfitting • We can get perfect classification
    performance on the training data by choosing a more complex model. • Complex models
    are tuned to the particular training samples, rather than on the characteristics
    of the true model. overfitting ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved 14 Generalization • Generalization is defined
    as the ability of a classifier to produce correct results on novel patterns. •
    How can we improve generalization performance ? • More training examples (i.e.,
    better model estimates). • Simpler models usually yield better performance. complex
    model simpler model ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of Singapore.
    All Rights Reserved 15 Confusion Matrix Actual class\Predicted class Predicted
    C Predicted ¬ C 1 1 Actual C True Positives (TP) False Negatives (FN) 1 Type-II
    Error Actual¬ C False Positives (FP) True Negatives (TN) 1 Type-I Error Accuracy
    = (TP + TN)/All Sensitivity = True Positive Rate = Recall= TP/(TP+FN) Specificity
    = True Negative Rate = TN/(FP+TN) Precision = TP/(TP+FP) F1 score = 2*Precision*
    Recall/(Precision + Recall) ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved 16 Actual class\Predicted class Predicted C
    1 Predicted ¬ C 1 Actual C 1 True Positives (TP) False Negatives (FN) Type-II
    Error Actual¬ C 1 False Positives (FP) Type-I Error True Negatives (TN) ROC (Receiver
    Operating Characteristic) Curve 100% 100% evitisoP evitisoP AUC = 90% eurT etaR
    eurT etaR AUC = 65% 0 0 % % [REDACTED_PHONE] False Positive % % % False Positive
    % Rate Rate • AUC = Area Under Curve • Overall measure of test performance • Comparisons
    between two tests based on differences between (estimated) AUC the higher the
    AUC, the better is the model. ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved 17 Gain Chart and Lift Chart ATA\S-PSUPR\Day1b.ppt\V3.0
    © 2024 National University of Singapore. All Rights Reserved 18 Hyperparameter
    Tuning • Hyperparameter are parameters that are not directly learnt within estimators.
    • For example, C, kernel and gamma for Support Vector machine. Learning rate,
    dropout rate, batch size, etc. for neural networks. • Methods used to find out
    Hyperparameters • Manual Search • Grid Search • Random Search • Bayesian Optimization
    • Evolutionary Optimization • … ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved [REDACTED_PHONE] Solving Pattern Recognition
    Problems Using Supervised Learning Techniques (I) ATA\S-PSUPR\Day1b.ppt\V3.0 ©
    2024 National University of Singapore. All Rights Reserved 20 Supervised Learning
    Techniques • Linear Regression & Logistic Regression • Instance based Learning
    (K-NN) • Naïve Bayes Classifiers • Decision Trees • Neural Networks • SVM and
    Kernel Methods ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of Singapore.
    All Rights Reserved 21 Linear Regression Use for numeric targets  Good if you
    know the target changes linearly  Assumes the model: t = ax +by +cz + d etc.
     income Minimises the sum of squared error age ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024
    National University of Singapore. All Rights Reserved 22 Linear Regression yˆ
    = βx +α i i y y i C A B y B A C y i *Least squares x estimation gave us the line
    (β) that minimized C2 n n n ∑ ( y − y) 2 = ∑ ( yˆ − y) 2 + ∑ ( yˆ − y ) 2 i i
    i i i=1 i=1 i=1 A2 B2 C2 R2=SS /SS reg total SS SS SS total reg residual Total
    squared distance of Distance from regression line to naïve Variance around the
    regression line observations from naïve mean mean of y of y Additional variability
    not explained Total variation Variability due to x (regression) by x—what least
    squares method aims to minimize ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved 23 B A2 B2 C2 SS total Total squared distance
    of observations from naïve mean of y Total variation SS reg Distance from regression
    line to naïve mean of y Variability due to x (regression) Logistic Regression
    Designed for classification problems • Tries to estimate class probabilities directly
    •  P  ln  =α+ βx + β x + ... + βx 1 1 2 2 i i 1− P  P= Class probability
    P/(1-P) = odds ln(p/(1-p) = logit (log odds) ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024
    National University of Singapore. All Rights Reserved 24 Logistic Regression •
    Logistic Regression For one input variable we can draw the logistic function as
    P(Target) 1 Input variable 0 Which is a good match for many T/F prediction situations
    The transformation ln(p/1-p) turns this into a straight line (p = prob(target))
    ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of Singapore. All Rights
    Reserved 25 Multinomial Logistic Regression • Generalizes logistic regression
    to multiclass problems. • Predict the probabilities of the different possible
    outcomes ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of Singapore. All
    Rights Reserved 26 K- Nearest Neighbour • Uses the “distances” between data items
    • E.g. Assign a new pattern to the most represented class in the K nearest neighbours
    (e.g. K = 5) Height High risk • Non-linear decision surfaces • Can be computationally
    intensive • Distance measure is important * Low risk Age What is the predicted
    class of the new pattern? ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved 27 K- Nearest Neighbour • Requires 3 things:
    • Feature Space(Training Data) • Distance metric • to compute distance between
    records • The value of k ? • the number of nearest neighbors to retrieve from
    which to get majority class • To classify an unknown record: • Compute distance
    to other training records • Identify k nearest neighbors • Use class labels of
    nearest neighbors to determine the class label of unknown record ATA\S-PSUPR\Day1b.ppt\V3.0
    © 2024 National University of Singapore. All Rights Reserved 28 K- Nearest Neighbour
    • Common Distance Metrics: • Euclidean distance • Hamming distance • Determine
    the class from k nearest neighbor list • Take the majority vote of class labels
    among the k-nearest neighbors • Weighted factor ICDM: Top Ten Data Mining Algorithms,
    k nearest neighbor classification, December 2006 29 ATA\S-PSUPR\Day1b.ppt\V3.0
    © 2024 National University of Singapore. All Rights Reserved 29 K- Nearest Neighbour
    • If k is too small, sensitive to noise points • If k is too large, neighborhood
    may include points from other classes • Choose an odd value for k, to eliminate
    ties k = 1:  ? Belongs to square class k = 3:  ? Belongs to triangle class ?
    k = 7:  ? Belongs to square class (Source: ICDM: Top Ten Data Mining Algorithms,
    k nearest neighbor classification,2006) ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National
    University of Singapore. All Rights Reserved 30 K- Nearest Neighbour Advantages
    • Simple technique that is easily implemented • Building model is inexpensive
    • Extremely flexible classification scheme • Nearest Neighbor classifiers are
    lazy learners • Scaling issues • Attributes may have to be scaled to prevent distance
    measures from being dominated by one of the attributes. ATA\S-PSUPR\Day1b.ppt\V3.0
    © 2024 National University of Singapore. All Rights Reserved 31 Naive Bayes •
    Naive Bayes is a probabilistic machine learning algorithm based on the Bayes Theorem.
    • It is used in a wide variety of classification tasks. • Typical applications
    include filtering spam, classifying documents, sentiment prediction, recommendation
    systems, etc. ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of Singapore.
    All Rights Reserved 32 Bayesian Classification • It performs probabilistic prediction,
    i.e., predicts class membership probabilities, based on Bayes’ Theorem. P(X|H)P(H)
    P(H |X) = P(X) • Let X be a data sample: class label is unknown • Let H be a hypothesis
    that X belongs to class Ci • Classification is to determine P(H|X), (i.e., posteriori
    probability): the probability that the hypothesis holds given the observed data
    sample X • P(H) (prior probability): the initial probability • P(X): probability
    that sample data is observed • P(X|H) (likelihood): the probability of observing
    the sample X, given that the hypothesis holds ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024
    National University of Singapore. All Rights Reserved 33 Bayesian Classification
    • Suppose X = (x , x , …, x ) and m classes C , C , …, C . 1 2 n 1 2 m • Classification
    is to derive the maximum posteriori, i.e., the maximal P(C |X) i • Using Bayes’
    theorem, maximize P(C |X) is equivalent to maximize i P(X|C )P(C ) i i P(X) •
    Since P(X) is constant for all classes, only P(X|C )P(C ) i i needs to be maximized
    ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of Singapore. All Rights
    Reserved 34 Naïve Bayes Classifier • A naïve Bayes classifier is a simple probabilistic
    classifier based on applying Bayes’ theorem with strong (naïve) independence assumptions.
    • It assumes that attributes are conditionally independent (i.e., no dependence
    relation between attributes): n P(X | C ) = ∏ P(x | C ) = P(x | C )× P(x | C )×...×
    P(x | C ) i i i i i k 1 2 n k =1 ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved 35 Naïve Bayes Classifier Example • Fruit Prediction
    Problem: predict if a given fruit is a ‘Banana’ or ‘Orange’ or ‘Other’ based on
    three features: long (0/1), sweet (0/1) and yellow (0/1). Fruit Long (x1) Sweet
    (x2) Yellow (x3) Training Orange 0 1 0 Data: Banana 1 0 1 Banana 1 1 1 Other 1
    1 0 … … ... … Source: https://www.machinelearningplus.com/predictive-modeling/how-naive-
    bayes-algorithm-works-with-example-and-full-code/ ATA\S-PSUPR\Day1b.ppt\V3.0 ©
    2024 National University of Singapore. All Rights Reserved 36 Fruit Long (x1)
    Sweet (x2) Yellow (x3) Orange 0 1 0 Banana 1 0 1 Banana 1 1 1 Other 1 1 0 … …
    ... … Naïve Bayes Classifier Example Let’s say you are given a fruit that is:
    Long (1), Sweet (1) and ? Yellow(1), can you predict what fruit it is ATA\S-PSUPR\Day1b.ppt\V3.0
    © 2024 National University of Singapore. All Rights Reserved 37 Naïve Bayes Classifier
    Example • Step 1: Compute the ‘Prior’ probabilities for each of the class of fruits.
    • P(C=Banana) = 500 / 1000 = 0.50 • P(C=Orange) = 300 / 1000 = 0.30 • P(C=Other)
    = 200 / 1000 = 0.20 ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of Singapore.
    All Rights Reserved 38 Naïve Bayes Classifier Example • Step 2: Compute the probability
    of evidence that goes in the denominator. (Optional) • P(x1=Long) = 500 / 1000
    = 0.50 • P(x2=Sweet) = 650 / 1000 = 0.65 • P(x3=Yellow) = 800 / 1000 = 0.80 •
    This is an optional step because the denominator is the same for all the classes
    and so will not affect the probabilities. ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National
    University of Singapore. All Rights Reserved 39 Naïve Bayes Classifier Example
    • Step 3: Compute the probability of likelihood of evidences that goes in the
    numerator. Probability of Likelihood for Banana: • P(x1=Long | C=Banana) = 400
    / 500 = 0.80 • P(x2=Sweet | C=Banana) = 350 / 500 = 0.70 • P(x3=Yellow | C=Banana)
    = 450 / 500 = 0.90 So, the overall probability of Likelihood of evidence for Banana
    = 0.8 * 0.7 * 0.9 = 0.504 ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved 40 Naïve Bayes Classifier Example • Step 4:
    Substitute all the values into the Naive Bayes formula to get the probability
    for “banana”. • P(C=Banana | X1=Long, X2=Sweet and X3=Yellow)= 𝑃𝑃 𝐿𝐿𝐿𝐿𝐿𝐿𝐿𝐿 𝐵𝐵𝐵𝐵𝐿𝐿𝐵𝐵𝐿𝐿𝐵𝐵
    ∗ 𝑃𝑃 𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆 𝐵𝐵𝐵𝐵𝐿𝐿𝐵𝐵𝐿𝐿𝐵𝐵) ∗ 𝑃𝑃(𝑌𝑌𝑆𝑆𝑌𝑌𝑌𝑌𝐿𝐿𝑆𝑆 𝐵𝐵𝐵𝐵𝐿𝐿𝐵𝐵𝐿𝐿𝐵𝐵 ∗ 𝑃𝑃(𝐵𝐵𝐵𝐵𝐿𝐿𝐵𝐵𝐿𝐿𝐵𝐵)
    𝑃𝑃 𝐿𝐿𝐿𝐿𝐿𝐿𝐿𝐿 ∗ 𝑃𝑃 𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆 ∗ 𝑃𝑃(𝑌𝑌𝑆𝑆𝑌𝑌𝑌𝑌𝐿𝐿𝑆𝑆) 0.8 ∗ 0.7 ∗ 0.9 ∗ 0.5 = = 0.252/𝑝𝑝(𝑆𝑆𝑒𝑒𝑒𝑒𝑒𝑒𝑆𝑆𝐿𝐿𝑒𝑒𝑆𝑆)
    𝑃𝑃(𝑆𝑆𝑒𝑒𝑒𝑒𝑒𝑒𝑆𝑆𝐿𝐿𝑒𝑒𝑆𝑆) ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of
    Singapore. All Rights Reserved 41 Naïve Bayes Classifier Example • Step 5: Repeat
    Step 3 and Step 4 to get the probability for “Orange” and “Other”. • P(C=Orange
    | X1=Long, X2=Sweet and X3=Yellow)= 𝑃𝑃 𝐿𝐿𝐿𝐿𝐿𝐿𝐿𝐿 𝑂𝑂𝑂𝑂𝐵𝐵𝐿𝐿𝐿𝐿𝑆𝑆 ∗ 𝑃𝑃 𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆 𝑂𝑂𝑂𝑂𝐵𝐵𝐿𝐿𝐿𝐿𝑆𝑆)
    ∗ 𝑃𝑃(𝑌𝑌𝑆𝑆𝑌𝑌𝑌𝑌𝐿𝐿𝑆𝑆 𝑂𝑂𝑂𝑂𝐵𝐵𝐿𝐿𝐿𝐿𝑆𝑆 ∗ 𝑃𝑃(𝑂𝑂𝑂𝑂𝐵𝐵𝐿𝐿𝐿𝐿𝑆𝑆) 𝑃𝑃 𝐿𝐿𝐿𝐿𝐿𝐿𝐿𝐿 ∗ 𝑃𝑃 𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆
    ∗ 𝑃𝑃(𝑌𝑌𝑆𝑆𝑌𝑌𝑌𝑌𝐿𝐿𝑆𝑆) [REDACTED_PHONE] ∗ ∗ ∗ [REDACTED_PHONE] = = 0/𝑝𝑝(𝑆𝑆𝑒𝑒𝑒𝑒𝑒𝑒𝑆𝑆𝐿𝐿𝑒𝑒𝑆𝑆)
    • P(C=Other | X1=Long, X2=Sweet and X3=Yellow)= 𝑃𝑃(𝑆𝑆𝑒𝑒𝑒𝑒𝑒𝑒𝑆𝑆𝐿𝐿𝑒𝑒𝑆𝑆) [REDACTED_PHONE]
    𝑃𝑃 ∗ 𝑃𝑃 𝑆𝑆𝑤𝑤𝑤𝑤𝑤𝑤𝑤𝑤 𝑂𝑂𝑤𝑤𝑂𝑤𝑤𝑂𝑂) ∗𝑃𝑃(𝑌𝑌𝑤𝑤𝑌𝑌𝑌𝑌𝑌𝑌𝑤𝑤 𝑂𝑂𝑤𝑤𝑂𝑤𝑤𝑂𝑂 ∗𝑃𝑃(𝑂𝑂𝑤𝑤𝑂𝑤𝑤𝑂𝑂) 200 ∗
    200 ∗ 200 ∗ 1000 𝐿𝐿𝐿𝐿𝐿𝐿𝐿𝐿 𝑂𝑂𝑆𝑆𝑂𝑆𝑆𝑂𝑂 𝑃𝑃 𝐿𝐿𝑌𝑌𝐿𝐿𝐿𝐿 ∗𝑃𝑃 𝑆𝑆𝑤𝑤𝑤𝑤𝑤𝑤𝑤𝑤 ∗𝑃𝑃(𝑌𝑌𝑤𝑤𝑌𝑌𝑌𝑌𝑌𝑌𝑤𝑤)
    = 𝑃𝑃(𝑤𝑤𝑒𝑒𝑒𝑒𝑑𝑑𝑤𝑤𝐿𝐿𝑒𝑒𝑤𝑤) = 0.01875/𝑝𝑝(𝑆𝑆𝑒𝑒𝑒𝑒𝑒𝑒𝑆𝑆𝐿𝐿𝑒𝑒𝑆𝑆) ATA\S-PSUPR\Day1b.ppt\V3.0
    © 2024 National University of Singapore. All Rights Reserved 42 Naïve Bayes Classifier
    Example Banana 0.252/𝑝𝑝(𝑆𝑆𝑒𝑒𝑒𝑒𝑒𝑒𝑆𝑆𝐿𝐿𝑒𝑒𝑆𝑆) {Long, Sweet, Orange Yellow} 0/𝑝𝑝(𝑆𝑆𝑒𝑒𝑒𝑒𝑒𝑒𝑆𝑆𝐿𝐿𝑒𝑒𝑆𝑆)
    Other 0.01875/𝑝𝑝(𝑆𝑆𝑒𝑒𝑒𝑒𝑒𝑒𝑆𝑆𝐿𝐿𝑒𝑒𝑆𝑆) Banana • Answer: – as it has the highest probability
    among the three classes. ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved 43 Laplace Correction • When having a model
    with many features, the entire probability will become zero because one of the
    feature’s value was zero, as we have seen in the previous example. • To avoid
    this, we increase the count of the variable with zero to a small value (usually
    1) in the numerator, so that the overall probability doesn’t become zero. ATA\S-PSUPR\Day1b.ppt\V3.0
    © 2024 National University of Singapore. All Rights Reserved 44 Pros and Cons
    of Naïve Bayes Classifier • Pros: • Easy and fast to predict. Perform well in
    multi class prediction • When assumption of independence holds, a Naive Bayes
    classifier performs better than other models like logistic regression and needs
    less training data. • It perform well in case of categorical input variables compared
    to numerical variables. For numerical variable, normal distribution is assumed.
    • Cons: • The assumption of independent predictors. In real life, it is almost
    impossible that we get a set of predictors which are completely independent. https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/
    ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of Singapore. All Rights
    Reserved 45 Modeling with Scikit-learn import numpyas np from sklearn.neighborsimport
    KNeighborsClassifier from sklearn.model_selectionimport train_test_split from
    sklearn.preprocessing import StandardScaler scaler = StandardScaler() from sklearn.datasetsimport
    load_iris scaler.fit(X_train) from sklearn.linear_modelimport LogisticRegression
    X_train= scaler.transform(X_train) from sklearn.metricsimport accuracy_score,classification_report,
    confusion_matrix X_test= scaler.transform(X_test) knn=KNeighborsClassifier(10)
    iris = load_iris() knn.fit(X_train, y_train) X = iris.data from sklearn.naive_bayes
    import GaussianNB y = iris.target nb=GaussianNB() X_train, X_test, y_train, y_test=
    train_test_split(X, y, random_state=0) nb.fit(X_train, y_train) LR = LogisticRegression(random_state=0,solver=''lbfgs'',multi_class=''multinomial'')
    predictions = nb.predict(X_test) LR.fit(X_train, y_train) print("Accuracy", accuracy_score(y_test,
    predictions)) print(confusion_matrix(y_test,predictions)) print(classification_report(y_test,predictions))
    ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of Singapore. All Rights
    Reserved [REDACTED_PHONE] WORKSHOP 1 SOLVING PATTERN RECOGNITION PROBLEMS USING
    PYTHON AND SCIKIT- LEARN ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved 47 Instructions • You may install and create
    your own Anaconda environment and run Jupyter Notebooks. • Alternatively, you
    can run Jupyter Notebooks using Google Colab. Note: To start working with Colab,
    you first need to log in to your Google account, then go to this link https://colab.research.google.com
    ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of Singapore. All Rights
    Reserved 48 Run Jupyter Notebook in Anaconda • Download and install latest Anaconda
    for Python 3.7: https://www.anaconda.com/download/ • Start Menu -> Anaconda Prompt:
    • conda create -n psupr python=3.7 • conda activate psupr • conda install numpy
    matplotlib jupyter pandas scikit-learn pydotplus • conda install -c conda-forge
    scikit-plot • Navigate to your working directory, eg. “d:\myfolder” • Run “jupyter
    notebook” • Now you can open .ipynb files within your browser ATA\S-PSUPR\Day1b.ppt\V3.0
    © 2024 National University of Singapore. All Rights Reserved 49 Problem Description
    Diabetes Prediction • This dataset is originally from the National Institute of
    Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to
    diagnostically predict whether or not a patient has diabetes, based on certain
    diagnostic measurements included in the dataset. • The datasets consists of several
    medical predictor variables and one target variable, Outcome. Predictor variables
    includes the number of pregnancies the patient has had, their BMI, insulin level,
    age, and so on. Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes,
    R.S. (1988).Using the ADAP learning algorithm to forecast the onset of diabetes
    mellitus.In Proceedings of the Symposium on Computer Applications and Medical
    Care(pp. 261--265). IEEE Computer Society Press. ATA\S-PSUPR\Day1b.ppt\V3.0 ©
    2024 National University of Singapore. All Rights Reserved 50 Workshop 1 • Open
    the jupyter notebook provided for this workshop. • As you go through the notebook,
    make sure you understand how each different model is built. (you can save notes
    as markdown in the notebook). • Compare the performance of these models. • Experiment
    with different parameter settings. • You may try with your own datasets. • Save
    your notebook with the cell output and upload it to Canvas. Last ATA\S-PSUPR\Day1b.ppt\V3.0
    © 2024 National University of Singapore. All Rights Reserved 51'
  - 'Accelerating Digital Excellence Copyright National University of SingaporeLive
    or Near-live Translation • Real time translation: start while sentence is spoken
    Source: S. Zhang et al, Wait-info Policy: Balancing Source and Target at Information
    Level for Simultaneous Machine Translation, EMNLP 2022. Agenda • Day 3 • 1: Speech
    processing basics • 2: Speech recognition (Speech-to-text) • 3: Case studies:
    Integrating speech recognition and NLP solutions • Day 4 • 4: Speech synthesis
    (Text-to-speech) • 5: Voice conversion and generation • 6: Spoken dialogue system
    (Spoken chatbot) 2 Copyright National University of Singapore Accelerating Digital
    Excellence ASR in Recent Years Source: D. Jurafskyand J.H. Martin, Speech and
    Language Processing. 46 Copyright National University of Singapore Accelerating
    Digital Excellence Evolution of AI systems AI systems perform ↑ better than humans
    ↓ AI systems perform worse Handwriting recognition Image recognition Language
    understanding Speech recognition Reading Reading comprehension comprehension Source:
    Kiela et al., Dynabench: Rethinking Benchmarking in NLP 48 Copyright National
    University of Singapore Accelerating Digital Excellence Evolution of ASR methods
    Source: Labellerr 49 Copyright National University of Singapore Accelerating Digital
    Excellence 75 A good tutorial: https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452
    Paradigm Shift • serves as a foundational layer upon which specialized applications
    are built. • involves extensive training data • being adapted to wide range of
    downstream tasks 78 Copyright National University of Singapore Accelerating Digital
    Excellence Whisper Model • Trained on 680k hours of multilingual and multitask
    supervised data • Three tasks: speech recognition, speech translation, and language
    recognition • #model parameters: 39M, 74M, 244M, 769M, 1550M 80 Source: https://openai.com/research/whisper
    Copyright National University of Singapore Accelerating Digital Excellence Wav2Vec2.0:
    Self Supervised Learning 81 Copyright National University of Singapore Accelerating
    Digital Excellence ChatGPT: LLM with Speech Capabilities 87 Copyright National
    University of Singapore Accelerating Digital Excellence Guidelines for Acoustic
    Modeling Units • Very phonetic languages (e.g., Spanish, German) • Letter-based
    or byte pair encoding (BPE) based acoustic modeling is effective. • Reasonably
    phonetic languages (e.g., English) • BPE-based acoustic model is preferred. •
    Letter-based acoustic modeling is also viable if large training datasets are available.
    • Non-phonetic or least phonetic languages (e.g., Chinese) • It is recommended
    to use a pronunciation dictionary to map words to modeling units. 90 Copyright
    National University of Singapore Accelerating Digital Excellence Further reading
    on Acoustic Modeling • Fine-Tune Whisper For Multilingual ASR with Transformers
    • https://huggingface.co/blog/fine-tune-whisper • Espnet: End-to-end speech processing
    • https://github.com/espnet/espnet • Kaldi: HMM-DNN acoustic modeling • https://kaldi-asr.org/
    • https://kaldi-asr.org/doc/tutorial.html • https://kaldi-asr.org/doc/kaldi_for_dummies.html
    • https://github.com/kaldi-asr/kaldi/tree/master/egs • HMM-based monophone and
    context-dependent triphone: • https://jonathan-hui.medium.com/speech-recognition-asr-model-training-
    90ed50d93615 91 Copyright National University of Singapore Accelerating Digital
    Excellence Further Reading on N-gram Language Models • Overview • https://web.eecs.umich.edu/~wangluxy/courses/eecs498_wn2021/slides_ee
    cs498_wn21/lm.pdf • LM training with SRILM • https://cmusphinx.github.io/wiki/tutoriallmadvanced/
    • LM linear interpolation/Building large n-gram LMs with SRILM • https://joshua.apache.org/6.0/large-lms.html
    • Morph n-gram model • http://research.spa.aalto.fi/speech/s895150/ex3.html 92
    Copyright National University of Singapore Accelerating Digital Excellence Open-source
    Python Speech Recognition • Installation: pip install SpeechRecognition • Library
    for performing speech recognition • Supported engines: • CMU Sphinx (works offline)
    • Google Speech Recognition • Google Cloud Speech API • Wit.ai • Microsoft Bing
    Voice Recognition • Houndify API • IBM Speech to Text • Snowboy Hotword Detection
    (works offline) • OpenAI Whisper (works offline) 107 Copyright National University
    of Singapore Accelerating Digital Excellence Open-source Python Speech Recognition
    • Recognize speech input from the microphone • Transcribe an audio file • Save
    audio data to an audio file • Calibrate the recognizer energy threshold for ambient
    noise levels • Listening to a microphone in the background • Website: https://pypi.org/project/SpeechRecognition/
    108 Copyright National University of Singapore Accelerating Digital Excellence
    Integrating speech recognition and NLP Possible applications: 1. Voice Search
    2. Customer Service Analysis 3. Speech Translation 4. Meeting Transcription and
    Summary Punctuation and capitalization Inverse text normalization restoration
    Copyright National University of Singapore Accelerating Digital Excellence Voice
    search • Find information using spoken queries, e.g. voice search in Taobao app
    • Text output from ASR often needs to transformed or normalized to match the input
    format expected by an existing text-based search system • e.g. iPhone sixteen
    sixty four g b => iPhone 16 64GB • Optimization for product terms (brand or product
    names) • Real me => Realme • Red me => Redmi Copyright National University of
    Singapore Accelerating Digital Excellence Quality Control of Customer Service
    • Customer and agent sentiment (from text and speech) • Speech sentiment analysis
    on voice characteristics such as pitch, loudness, etc • Text sentiment analysis
    • Non-talk time • Talk speed • Interruptions – Still challenging! Copyright National
    University of Singapore Accelerating Digital Excellence Speech Translation • Applications
    of machine translation include: • Cross-border communication • Localization of
    websites (and other digit content) • Language learning • Speech translation enables
    users to watch foreign videos, such as films and lectures, in their own language
    Source: https://www.mdpi.com/[REDACTED_PHONE]/13/15/8900 Copyright National University
    of Singapore Accelerating Digital Excellence Evolution of Machine Translation
    Methods Source: https://medium.com/free-code-camp/a-history-of-machine-translation-from-the-cold-war-to-deep-learning-f1d335ce8b5
    118 Copyright National University of Singapore Accelerating Digital Excellence
    Evolution of Machine Translation Methods 119 Copyright National University of
    Singapore Accelerating Digital Excellence Evaluation Metrics • Bilingual Evaluation
    Understudy (BLEU) Score • Calculated by comparing n-grams of machine-translated
    sentences to those of human-translated sentences. • Higher scores represent better
    MT performances. • BLEU scores may decreases as sentence lengths increase 120
    Copyright National University of Singapore Accelerating Digital Excellence Catastrophic
    Errors in MT • Generation of profanity • Eliminate words that appear in language-specific
    offensive word list. • BUT offensive language is not limited to specific words
    • Generation of violent or inciting content. • Reversal of intended meaning. •
    Mistranslation of proper names. 121 Copyright National University of Singapore
    Accelerating Digital Excellence Practical Challenges in Speech Translation • Misrecognized
    words lead to inaccurate or nonsensical translations. • Out-of-vocabulary words
    and domain-specific terms can be mistranslated or omitted. • MT systems for low-resource
    languages often exhibit poor accuracy, limiting their real-world utility. • Hesitations,
    false starts and filler words (“um”, “uh”) in ASR output lead to awkward or inaccurate
    MT output. • Inaccurate segmentation can confuse the MT model and reduce translation
    quality. • Live or near-live translation requires low latency. This can conflict
    with the need for high accuracy. 122 Copyright National University of Singapore
    Accelerating Digital Excellence Live or Near-live Translation • Real time translation:
    start while sentence is spoken I am going to talk today about energy and climate.
    Heute spreche ich zu Ihnen über Energie und Klima. • Subtitles: have to be readable
    in limited time • Dubbing: sync up with video of speaker’s mouth movement Copyright
    National University of Singapore Accelerating Digital Excellence Live or Near-live
    Translation • Real time translation: start while sentence is spoken Source: R.
    Zhang et al, Dynamic Sentence Boundary Detection for Simultaneous Translation,
    Proceedings of the 1st Workshop on Automatic Simultaneous Translation. Copyright
    National University of Singapore Accelerating Digital Excellence Live or Near-live
    Translation • Real time translation: start while sentence is spoken Source: S.
    Zhang et al, Wait-info Policy: Balancing Source and Target at Information Level
    for Simultaneous Machine Translation, EMNLP 2022. Copyright National University
    of Singapore Accelerating Digital Excellence Handling Context Across Multiple
    Sentences • Translation of pronouns may require co-reference 129 Copyright National
    University of Singapore Accelerating Digital Excellence Meeting Transcription
    Otter.ai Copyright National University of Singapore Accelerating Digital Excellence
    Meeting Summarization Evaluation metrics for summarization: BLEU, ROUGE Copyright
    National University of Singapore Accelerating Digital Excellence Practical Challenges
    in Meeting Summarization • Misrecognized words lead to inaccurate summary. • Business,
    medical, or legal meetings contain domain-specific terms that need accurate recognition
    and summarization. • Hesitations, false starts and filler words (“um”, “uh”) in
    ASR output lead to awkward or inaccurate summary. • Identifying and differentiating
    speakers correctly is crucial for attributing statements accurately. • Proper
    co-reference resolution of pronouns and vague references (this, that, those things)
    is important for accurate attribution of actions and decisions. • Different expectations
    of what is ”important” in a meeting. 132 Copyright National University of Singapore
    Accelerating Digital Excellence Infrastructure (GPU) Consideration for Production
    • Find something that fits your budget • Key performance metrics to be considered:
    • Token Per Second (TPS) = (Input Tokens + Output Tokens) / Total Turnaround Time
    • Time To First Token (TTFT) • GPU Usage: starting with T4 16GB GPU that works
    well for 3B or 8B models • Consider quantization and distillation of models (i.e.
    smaller models) • Consider small models with few-shot prompting. Source: https://rumn.medium.com/benchmarking-llm-performance-token-per-second-tps-time-to-first-token-ttft-and-gpu-usage-8c50ee8387fa
    https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html https://medium.com/aimonks/what-is-quantization-and-distillation-of-models-a67e3a2dc325
    138 Copyright National University of Singapore Accelerating Digital Excellence
    Gartner Hype Cycle 139 Copyright National University of Singapore Accelerating
    Digital Excellence'
  - 'ATA\ S-PSUPR \Day1b.ppt \V3.0 © 2024 National University of Singapore. All Rights
    Reserved 21 Supervised Learning Techniques •Linear Regression & Logistic Regression
    •Instance based Learning (K -NN) •Naïve Bayes Classifiers •Decision Trees •Neural
    Networks •SVM and Kernel Methods PROBLEM SOLVING USING PATTERN RECOGNITION DAY
    1B Dr Zhu Fangming NUS-ISS National University of Singapore [REDACTED_EMAIL] Not
    be reproduced in any form or by any means, without the written permission of ISS,
    NUS, other than for the purpose for which it has been supplied. ATA\S-PSUPR\Day1b.ppt\V3.0
    © 2024 National University of Singapore. All Rights Reserved [REDACTED_PHONE]
    HOW TO ANALYSE, MODEL AND SOLVE PATTERN RECOGNITION PROBLEMS ATA\S-PSUPR\Day1b.ppt\V3.0
    © 2024 National University of Singapore. All Rights Reserved 2 Topics • Important
    steps in solving pattern recognition problems • Important issues for pattern recognition:
    data pre- processing, feature selection, model evaluation. ATA\S-PSUPR\Day1b.ppt\V3.0
    © 2024 National University of Singapore. All Rights Reserved 3 Pattern Recognition
    Process with Supervised Learning ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved 4 Models, Features and Classes • A pattern is
    represented by a set of d features, or attributes, viewed as a d-dimensional feature
    vector. , )T 𝟏𝟏 𝟐𝟐 𝑑𝑑 𝑿𝑿 = (𝒙𝒙 𝒙𝒙 , … 𝒙𝒙 x: input vector Classification / y: class
    label (pattern with Regression /regression features) Model p(x,y) How do we model
    p(x,y)? ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of Singapore. All
    Rights Reserved 5 Data Pre-processing  Data cleaning Fill in missing values,
    smooth noisy data, identify or remove outliers, and • resolve inconsistencies
     Data integration Integration of multiple databases •  Data transformation Normalization
    and aggregation •  Data reduction Dimensionality reduction - feature selection
    • Numerosity reduction – select/ sample records • ATA\S-PSUPR\Day1b.ppt\V3.0 ©
    2024 National University of Singapore. All Rights Reserved 6 Normalization • Normalization
    & feature scaling techniques are important for many machine learning algorithms.
    • Min-Max scaling [0,1] • Z-score (standardization) • Use the same parameters
    on the test dataset and new unseen data. ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National
    University of Singapore. All Rights Reserved 7 Feature Selection • Curse of dimensionality
    • Retain only "useful" (discriminatory) information and avoid overfitting. • Reasons
    to reduce the number of features: • Computational complexity • Generalization
    properties ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of Singapore.
    All Rights Reserved 8 Dimensionality Reduction • Principal Component Analyses
    (PCA) and Linear Discriminant Analysis (LDA) can be used. • Linear Discriminant
    Analysis (LDA) tries to identify attributes that account for the most variance
    between classes. • In particular, LDA, in contrast to PCA, is a supervised method,
    using known class labels. ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved 9 Data Partition and Preparation Training set
    vs. test set vs. validation set Cross-validation https://www.datasciencecentral.com
    ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of Singapore. All Rights
    Reserved 10 Cross Validation ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved 11 Learning from Imbalanced Data • Techniques
    for Learning from imbalanced data: • Data Augmentation • Custom Loss Function
    Z=0 Z=1 • Fraud detection • Churn Modeling • Anomaly detection ATA\S-PSUPR\Day1b.ppt\V3.0
    © 2024 National University of Singapore. All Rights Reserved 12 Model Evaluation
    • Error Measures • Overtraining/overfitting • Confusion Matrix • ROC Charts •
    Gains Chart/ Lift Chart ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved 13 Overfitting • We can get perfect classification
    performance on the training data by choosing a more complex model. • Complex models
    are tuned to the particular training samples, rather than on the characteristics
    of the true model. overfitting ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved 14 Generalization • Generalization is defined
    as the ability of a classifier to produce correct results on novel patterns. •
    How can we improve generalization performance ? • More training examples (i.e.,
    better model estimates). • Simpler models usually yield better performance. complex
    model simpler model ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of Singapore.
    All Rights Reserved 15 Confusion Matrix Actual class\Predicted class Predicted
    C Predicted ¬ C 1 1 Actual C True Positives (TP) False Negatives (FN) 1 Type-II
    Error Actual¬ C False Positives (FP) True Negatives (TN) 1 Type-I Error Accuracy
    = (TP + TN)/All Sensitivity = True Positive Rate = Recall= TP/(TP+FN) Specificity
    = True Negative Rate = TN/(FP+TN) Precision = TP/(TP+FP) F1 score = 2*Precision*
    Recall/(Precision + Recall) ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved 16 Actual class\Predicted class Predicted C
    1 Predicted ¬ C 1 Actual C 1 True Positives (TP) False Negatives (FN) Type-II
    Error Actual¬ C 1 False Positives (FP) Type-I Error True Negatives (TN) ROC (Receiver
    Operating Characteristic) Curve 100% 100% evitisoP evitisoP AUC = 90% eurT etaR
    eurT etaR AUC = 65% 0 0 % % [REDACTED_PHONE] False Positive % % % False Positive
    % Rate Rate • AUC = Area Under Curve • Overall measure of test performance • Comparisons
    between two tests based on differences between (estimated) AUC the higher the
    AUC, the better is the model. ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved 17 Gain Chart and Lift Chart ATA\S-PSUPR\Day1b.ppt\V3.0
    © 2024 National University of Singapore. All Rights Reserved 18 Hyperparameter
    Tuning • Hyperparameter are parameters that are not directly learnt within estimators.
    • For example, C, kernel and gamma for Support Vector machine. Learning rate,
    dropout rate, batch size, etc. for neural networks. • Methods used to find out
    Hyperparameters • Manual Search • Grid Search • Random Search • Bayesian Optimization
    • Evolutionary Optimization • … ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved [REDACTED_PHONE] Solving Pattern Recognition
    Problems Using Supervised Learning Techniques (I) ATA\S-PSUPR\Day1b.ppt\V3.0 ©
    2024 National University of Singapore. All Rights Reserved 20 Supervised Learning
    Techniques • Linear Regression & Logistic Regression • Instance based Learning
    (K-NN) • Naïve Bayes Classifiers • Decision Trees • Neural Networks • SVM and
    Kernel Methods ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of Singapore.
    All Rights Reserved 21 Linear Regression Use for numeric targets  Good if you
    know the target changes linearly  Assumes the model: t = ax +by +cz + d etc.
     income Minimises the sum of squared error age ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024
    National University of Singapore. All Rights Reserved 22 Linear Regression yˆ
    = βx +α i i y y i C A B y B A C y i *Least squares x estimation gave us the line
    (β) that minimized C2 n n n ∑ ( y − y) 2 = ∑ ( yˆ − y) 2 + ∑ ( yˆ − y ) 2 i i
    i i i=1 i=1 i=1 A2 B2 C2 R2=SS /SS reg total SS SS SS total reg residual Total
    squared distance of Distance from regression line to naïve Variance around the
    regression line observations from naïve mean mean of y of y Additional variability
    not explained Total variation Variability due to x (regression) by x—what least
    squares method aims to minimize ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved 23 B A2 B2 C2 SS total Total squared distance
    of observations from naïve mean of y Total variation SS reg Distance from regression
    line to naïve mean of y Variability due to x (regression) Logistic Regression
    Designed for classification problems • Tries to estimate class probabilities directly
    •  P  ln  =α+ βx + β x + ... + βx 1 1 2 2 i i 1− P  P= Class probability
    P/(1-P) = odds ln(p/(1-p) = logit (log odds) ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024
    National University of Singapore. All Rights Reserved 24 Logistic Regression •
    Logistic Regression For one input variable we can draw the logistic function as
    P(Target) 1 Input variable 0 Which is a good match for many T/F prediction situations
    The transformation ln(p/1-p) turns this into a straight line (p = prob(target))
    ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of Singapore. All Rights
    Reserved 25 Multinomial Logistic Regression • Generalizes logistic regression
    to multiclass problems. • Predict the probabilities of the different possible
    outcomes ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of Singapore. All
    Rights Reserved 26 K- Nearest Neighbour • Uses the “distances” between data items
    • E.g. Assign a new pattern to the most represented class in the K nearest neighbours
    (e.g. K = 5) Height High risk • Non-linear decision surfaces • Can be computationally
    intensive • Distance measure is important * Low risk Age What is the predicted
    class of the new pattern? ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved 27 K- Nearest Neighbour • Requires 3 things:
    • Feature Space(Training Data) • Distance metric • to compute distance between
    records • The value of k ? • the number of nearest neighbors to retrieve from
    which to get majority class • To classify an unknown record: • Compute distance
    to other training records • Identify k nearest neighbors • Use class labels of
    nearest neighbors to determine the class label of unknown record ATA\S-PSUPR\Day1b.ppt\V3.0
    © 2024 National University of Singapore. All Rights Reserved 28 K- Nearest Neighbour
    • Common Distance Metrics: • Euclidean distance • Hamming distance • Determine
    the class from k nearest neighbor list • Take the majority vote of class labels
    among the k-nearest neighbors • Weighted factor ICDM: Top Ten Data Mining Algorithms,
    k nearest neighbor classification, December 2006 29 ATA\S-PSUPR\Day1b.ppt\V3.0
    © 2024 National University of Singapore. All Rights Reserved 29 K- Nearest Neighbour
    • If k is too small, sensitive to noise points • If k is too large, neighborhood
    may include points from other classes • Choose an odd value for k, to eliminate
    ties k = 1:  ? Belongs to square class k = 3:  ? Belongs to triangle class ?
    k = 7:  ? Belongs to square class (Source: ICDM: Top Ten Data Mining Algorithms,
    k nearest neighbor classification,2006) ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National
    University of Singapore. All Rights Reserved 30 K- Nearest Neighbour Advantages
    • Simple technique that is easily implemented • Building model is inexpensive
    • Extremely flexible classification scheme • Nearest Neighbor classifiers are
    lazy learners • Scaling issues • Attributes may have to be scaled to prevent distance
    measures from being dominated by one of the attributes. ATA\S-PSUPR\Day1b.ppt\V3.0
    © 2024 National University of Singapore. All Rights Reserved 31 Naive Bayes •
    Naive Bayes is a probabilistic machine learning algorithm based on the Bayes Theorem.
    • It is used in a wide variety of classification tasks. • Typical applications
    include filtering spam, classifying documents, sentiment prediction, recommendation
    systems, etc. ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of Singapore.
    All Rights Reserved 32 Bayesian Classification • It performs probabilistic prediction,
    i.e., predicts class membership probabilities, based on Bayes’ Theorem. P(X|H)P(H)
    P(H |X) = P(X) • Let X be a data sample: class label is unknown • Let H be a hypothesis
    that X belongs to class Ci • Classification is to determine P(H|X), (i.e., posteriori
    probability): the probability that the hypothesis holds given the observed data
    sample X • P(H) (prior probability): the initial probability • P(X): probability
    that sample data is observed • P(X|H) (likelihood): the probability of observing
    the sample X, given that the hypothesis holds ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024
    National University of Singapore. All Rights Reserved 33 Bayesian Classification
    • Suppose X = (x , x , …, x ) and m classes C , C , …, C . 1 2 n 1 2 m • Classification
    is to derive the maximum posteriori, i.e., the maximal P(C |X) i • Using Bayes’
    theorem, maximize P(C |X) is equivalent to maximize i P(X|C )P(C ) i i P(X) •
    Since P(X) is constant for all classes, only P(X|C )P(C ) i i needs to be maximized
    ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of Singapore. All Rights
    Reserved 34 Naïve Bayes Classifier • A naïve Bayes classifier is a simple probabilistic
    classifier based on applying Bayes’ theorem with strong (naïve) independence assumptions.
    • It assumes that attributes are conditionally independent (i.e., no dependence
    relation between attributes): n P(X | C ) = ∏ P(x | C ) = P(x | C )× P(x | C )×...×
    P(x | C ) i i i i i k 1 2 n k =1 ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved 35 Naïve Bayes Classifier Example • Fruit Prediction
    Problem: predict if a given fruit is a ‘Banana’ or ‘Orange’ or ‘Other’ based on
    three features: long (0/1), sweet (0/1) and yellow (0/1). Fruit Long (x1) Sweet
    (x2) Yellow (x3) Training Orange 0 1 0 Data: Banana 1 0 1 Banana 1 1 1 Other 1
    1 0 … … ... … Source: https://www.machinelearningplus.com/predictive-modeling/how-naive-
    bayes-algorithm-works-with-example-and-full-code/ ATA\S-PSUPR\Day1b.ppt\V3.0 ©
    2024 National University of Singapore. All Rights Reserved 36 Fruit Long (x1)
    Sweet (x2) Yellow (x3) Orange 0 1 0 Banana 1 0 1 Banana 1 1 1 Other 1 1 0 … …
    ... … Naïve Bayes Classifier Example Let’s say you are given a fruit that is:
    Long (1), Sweet (1) and ? Yellow(1), can you predict what fruit it is ATA\S-PSUPR\Day1b.ppt\V3.0
    © 2024 National University of Singapore. All Rights Reserved 37 Naïve Bayes Classifier
    Example • Step 1: Compute the ‘Prior’ probabilities for each of the class of fruits.
    • P(C=Banana) = 500 / 1000 = 0.50 • P(C=Orange) = 300 / 1000 = 0.30 • P(C=Other)
    = 200 / 1000 = 0.20 ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of Singapore.
    All Rights Reserved 38 Naïve Bayes Classifier Example • Step 2: Compute the probability
    of evidence that goes in the denominator. (Optional) • P(x1=Long) = 500 / 1000
    = 0.50 • P(x2=Sweet) = 650 / 1000 = 0.65 • P(x3=Yellow) = 800 / 1000 = 0.80 •
    This is an optional step because the denominator is the same for all the classes
    and so will not affect the probabilities. ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National
    University of Singapore. All Rights Reserved 39 Naïve Bayes Classifier Example
    • Step 3: Compute the probability of likelihood of evidences that goes in the
    numerator. Probability of Likelihood for Banana: • P(x1=Long | C=Banana) = 400
    / 500 = 0.80 • P(x2=Sweet | C=Banana) = 350 / 500 = 0.70 • P(x3=Yellow | C=Banana)
    = 450 / 500 = 0.90 So, the overall probability of Likelihood of evidence for Banana
    = 0.8 * 0.7 * 0.9 = 0.504 ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved 40 Naïve Bayes Classifier Example • Step 4:
    Substitute all the values into the Naive Bayes formula to get the probability
    for “banana”. • P(C=Banana | X1=Long, X2=Sweet and X3=Yellow)= 𝑃𝑃 𝐿𝐿𝐿𝐿𝐿𝐿𝐿𝐿 𝐵𝐵𝐵𝐵𝐿𝐿𝐵𝐵𝐿𝐿𝐵𝐵
    ∗ 𝑃𝑃 𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆 𝐵𝐵𝐵𝐵𝐿𝐿𝐵𝐵𝐿𝐿𝐵𝐵) ∗ 𝑃𝑃(𝑌𝑌𝑆𝑆𝑌𝑌𝑌𝑌𝐿𝐿𝑆𝑆 𝐵𝐵𝐵𝐵𝐿𝐿𝐵𝐵𝐿𝐿𝐵𝐵 ∗ 𝑃𝑃(𝐵𝐵𝐵𝐵𝐿𝐿𝐵𝐵𝐿𝐿𝐵𝐵)
    𝑃𝑃 𝐿𝐿𝐿𝐿𝐿𝐿𝐿𝐿 ∗ 𝑃𝑃 𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆 ∗ 𝑃𝑃(𝑌𝑌𝑆𝑆𝑌𝑌𝑌𝑌𝐿𝐿𝑆𝑆) 0.8 ∗ 0.7 ∗ 0.9 ∗ 0.5 = = 0.252/𝑝𝑝(𝑆𝑆𝑒𝑒𝑒𝑒𝑒𝑒𝑆𝑆𝐿𝐿𝑒𝑒𝑆𝑆)
    𝑃𝑃(𝑆𝑆𝑒𝑒𝑒𝑒𝑒𝑒𝑆𝑆𝐿𝐿𝑒𝑒𝑆𝑆) ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of
    Singapore. All Rights Reserved 41 Naïve Bayes Classifier Example • Step 5: Repeat
    Step 3 and Step 4 to get the probability for “Orange” and “Other”. • P(C=Orange
    | X1=Long, X2=Sweet and X3=Yellow)= 𝑃𝑃 𝐿𝐿𝐿𝐿𝐿𝐿𝐿𝐿 𝑂𝑂𝑂𝑂𝐵𝐵𝐿𝐿𝐿𝐿𝑆𝑆 ∗ 𝑃𝑃 𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆 𝑂𝑂𝑂𝑂𝐵𝐵𝐿𝐿𝐿𝐿𝑆𝑆)
    ∗ 𝑃𝑃(𝑌𝑌𝑆𝑆𝑌𝑌𝑌𝑌𝐿𝐿𝑆𝑆 𝑂𝑂𝑂𝑂𝐵𝐵𝐿𝐿𝐿𝐿𝑆𝑆 ∗ 𝑃𝑃(𝑂𝑂𝑂𝑂𝐵𝐵𝐿𝐿𝐿𝐿𝑆𝑆) 𝑃𝑃 𝐿𝐿𝐿𝐿𝐿𝐿𝐿𝐿 ∗ 𝑃𝑃 𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆
    ∗ 𝑃𝑃(𝑌𝑌𝑆𝑆𝑌𝑌𝑌𝑌𝐿𝐿𝑆𝑆) [REDACTED_PHONE] ∗ ∗ ∗ [REDACTED_PHONE] = = 0/𝑝𝑝(𝑆𝑆𝑒𝑒𝑒𝑒𝑒𝑒𝑆𝑆𝐿𝐿𝑒𝑒𝑆𝑆)
    • P(C=Other | X1=Long, X2=Sweet and X3=Yellow)= 𝑃𝑃(𝑆𝑆𝑒𝑒𝑒𝑒𝑒𝑒𝑆𝑆𝐿𝐿𝑒𝑒𝑆𝑆) [REDACTED_PHONE]
    𝑃𝑃 ∗ 𝑃𝑃 𝑆𝑆𝑤𝑤𝑤𝑤𝑤𝑤𝑤𝑤 𝑂𝑂𝑤𝑤𝑂𝑤𝑤𝑂𝑂) ∗𝑃𝑃(𝑌𝑌𝑤𝑤𝑌𝑌𝑌𝑌𝑌𝑌𝑤𝑤 𝑂𝑂𝑤𝑤𝑂𝑤𝑤𝑂𝑂 ∗𝑃𝑃(𝑂𝑂𝑤𝑤𝑂𝑤𝑤𝑂𝑂) 200 ∗
    200 ∗ 200 ∗ 1000 𝐿𝐿𝐿𝐿𝐿𝐿𝐿𝐿 𝑂𝑂𝑆𝑆𝑂𝑆𝑆𝑂𝑂 𝑃𝑃 𝐿𝐿𝑌𝑌𝐿𝐿𝐿𝐿 ∗𝑃𝑃 𝑆𝑆𝑤𝑤𝑤𝑤𝑤𝑤𝑤𝑤 ∗𝑃𝑃(𝑌𝑌𝑤𝑤𝑌𝑌𝑌𝑌𝑌𝑌𝑤𝑤)
    = 𝑃𝑃(𝑤𝑤𝑒𝑒𝑒𝑒𝑑𝑑𝑤𝑤𝐿𝐿𝑒𝑒𝑤𝑤) = 0.01875/𝑝𝑝(𝑆𝑆𝑒𝑒𝑒𝑒𝑒𝑒𝑆𝑆𝐿𝐿𝑒𝑒𝑆𝑆) ATA\S-PSUPR\Day1b.ppt\V3.0
    © 2024 National University of Singapore. All Rights Reserved 42 Naïve Bayes Classifier
    Example Banana 0.252/𝑝𝑝(𝑆𝑆𝑒𝑒𝑒𝑒𝑒𝑒𝑆𝑆𝐿𝐿𝑒𝑒𝑆𝑆) {Long, Sweet, Orange Yellow} 0/𝑝𝑝(𝑆𝑆𝑒𝑒𝑒𝑒𝑒𝑒𝑆𝑆𝐿𝐿𝑒𝑒𝑆𝑆)
    Other 0.01875/𝑝𝑝(𝑆𝑆𝑒𝑒𝑒𝑒𝑒𝑒𝑆𝑆𝐿𝐿𝑒𝑒𝑆𝑆) Banana • Answer: – as it has the highest probability
    among the three classes. ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved 43 Laplace Correction • When having a model
    with many features, the entire probability will become zero because one of the
    feature’s value was zero, as we have seen in the previous example. • To avoid
    this, we increase the count of the variable with zero to a small value (usually
    1) in the numerator, so that the overall probability doesn’t become zero. ATA\S-PSUPR\Day1b.ppt\V3.0
    © 2024 National University of Singapore. All Rights Reserved 44 Pros and Cons
    of Naïve Bayes Classifier • Pros: • Easy and fast to predict. Perform well in
    multi class prediction • When assumption of independence holds, a Naive Bayes
    classifier performs better than other models like logistic regression and needs
    less training data. • It perform well in case of categorical input variables compared
    to numerical variables. For numerical variable, normal distribution is assumed.
    • Cons: • The assumption of independent predictors. In real life, it is almost
    impossible that we get a set of predictors which are completely independent. https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/
    ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of Singapore. All Rights
    Reserved 45 Modeling with Scikit-learn import numpyas np from sklearn.neighborsimport
    KNeighborsClassifier from sklearn.model_selectionimport train_test_split from
    sklearn.preprocessing import StandardScaler scaler = StandardScaler() from sklearn.datasetsimport
    load_iris scaler.fit(X_train) from sklearn.linear_modelimport LogisticRegression
    X_train= scaler.transform(X_train) from sklearn.metricsimport accuracy_score,classification_report,
    confusion_matrix X_test= scaler.transform(X_test) knn=KNeighborsClassifier(10)
    iris = load_iris() knn.fit(X_train, y_train) X = iris.data from sklearn.naive_bayes
    import GaussianNB y = iris.target nb=GaussianNB() X_train, X_test, y_train, y_test=
    train_test_split(X, y, random_state=0) nb.fit(X_train, y_train) LR = LogisticRegression(random_state=0,solver=''lbfgs'',multi_class=''multinomial'')
    predictions = nb.predict(X_test) LR.fit(X_train, y_train) print("Accuracy", accuracy_score(y_test,
    predictions)) print(confusion_matrix(y_test,predictions)) print(classification_report(y_test,predictions))
    ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of Singapore. All Rights
    Reserved [REDACTED_PHONE] WORKSHOP 1 SOLVING PATTERN RECOGNITION PROBLEMS USING
    PYTHON AND SCIKIT- LEARN ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University
    of Singapore. All Rights Reserved 47 Instructions • You may install and create
    your own Anaconda environment and run Jupyter Notebooks. • Alternatively, you
    can run Jupyter Notebooks using Google Colab. Note: To start working with Colab,
    you first need to log in to your Google account, then go to this link https://colab.research.google.com
    ATA\S-PSUPR\Day1b.ppt\V3.0 © 2024 National University of Singapore. All Rights
    Reserved 48 Run Jupyter Notebook in Anaconda • Download and install latest Anaconda
    for Python 3.7: https://www.anaconda.com/download/ • Start Menu -> Anaconda Prompt:
    • conda create -n psupr python=3.7 • conda activate psupr • conda install numpy
    matplotlib jupyter pandas scikit-learn pydotplus • conda install -c conda-forge
    scikit-plot • Navigate to your working directory, eg. “d:\myfolder” • Run “jupyter
    notebook” • Now you can open .ipynb files within your browser ATA\S-PSUPR\Day1b.ppt\V3.0
    © 2024 National University of Singapore. All Rights Reserved 49 Problem Description
    Diabetes Prediction • This dataset is originally from the National Institute of
    Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to
    diagnostically predict whether or not a patient has diabetes, based on certain
    diagnostic measurements included in the dataset. • The datasets consists of several
    medical predictor variables and one target variable, Outcome. Predictor variables
    includes the number of pregnancies the patient has had, their BMI, insulin level,
    age, and so on. Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes,
    R.S. (1988).Using the ADAP learning algorithm to forecast the onset of diabetes
    mellitus.In Proceedings of the Symposium on Computer Applications and Medical
    Care(pp. 261--265). IEEE Computer Society Press. ATA\S-PSUPR\Day1b.ppt\V3.0 ©
    2024 National University of Singapore. All Rights Reserved 50 Workshop 1 • Open
    the jupyter notebook provided for this workshop. • As you go through the notebook,
    make sure you understand how each different model is built. (you can save notes
    as markdown in the notebook). • Compare the performance of these models. • Experiment
    with different parameter settings. • You may try with your own datasets. • Save
    your notebook with the cell output and upload it to Canvas. Last ATA\S-PSUPR\Day1b.ppt\V3.0
    © 2024 National University of Singapore. All Rights Reserved 51'
- source_sentence: Dear Participant Greetings from NUS -ISS! We are honoured to partner
    you in your lifelong learning journey, as you climb the next curve of digital
    excellence. Our aim is to provide you with the necessary skills and knowledge
    to aid in your professional and personal growth. Please be advised that the course
    materials, including the course notes, are protected by copyright and are property
    of the National University of Singapore or t heir respective copyright owners.
    The course notes provided are for personal use and reference during the course
    delivery, and any broadcasting, dissemination, distribution, preparation, reproduction,
    sub- licensing, transmission, uploading, or of derivativ e works is strictly prohibited.
    NUS -ISS is dedicated to supporting your growth through a comprehensive portfolio
    of multiple learning pathways, with a wide spectrum of programmes in critical
    industry disciplines, including artificial intelligence, cybersecurity, data science,
    digital governance, digital innovation, smart health, software development and
    more. Should you require any assistance or information regarding our blended learning
    programmes, executive programmes, graduate and stackable programmes , please do
    not hesitate to reach out to us at ask [REDACTED_EMAIL] or via phone at ([REDACTED_PHONE].
    Wishing you a productive and fulfilling learning experience. Kind Regards, Yum
    Hui Yuen (Mrs) Director Executive Education & Corporate Services NUS -ISS National
    University of Singapore
  sentences:
  - '© 2025 National University of Singapore. All Rights Reserved Speaker Diarization
    •Determine “who spoke when ?” •Segment and cluster the audio such that all segments
    belonging to a particular speaker are grouped together . Page 66 Reference: https://docs.nvidia.com/deeplearning/nemo/user
    -guide/docs/en/main/asr/speaker_diarization/intro.html'
  - Dear Participant Greetings from NUS -ISS! We are honoured to partner you in your
    lifelong learning journey, as you climb the next curve of digital excellence.
    Our aim is to provide you with the necessary skills and knowledge to aid in your
    professional and personal growth. Please be advised that the course materials,
    including the course notes, are protected by copyright and are property of the
    National University of Singapore or t heir respective copyright owners. The course
    notes provided are for personal use and reference during the course delivery,
    and any broadcasting, dissemination, distribution, preparation, reproduction,
    sub- licensing, transmission, uploading, or of derivativ e works is strictly prohibited.
    NUS -ISS is dedicated to supporting your growth through a comprehensive portfolio
    of multiple learning pathways, with a wide spectrum of programmes in critical
    industry disciplines, including artificial intelligence, cybersecurity, data science,
    digital governance, digital innovation, smart health, software development and
    more. Should you require any assistance or information regarding our blended learning
    programmes, executive programmes, graduate and stackable programmes , please do
    not hesitate to reach out to us at ask [REDACTED_EMAIL] or via phone at ([REDACTED_PHONE].
    Wishing you a productive and fulfilling learning experience. Kind Regards, Yum
    Hui Yuen (Mrs) Director Executive Education & Corporate Services NUS -ISS National
    University of Singapore
  - 'Master of Technology in Intelligent Systems page 14 of 25 MTech IS Grad Cert
    Exam: Intelligent Reasoning Systems Figure Q2.8: A-EYE-Web: Diagnosis and Recommendation
    (single diagnosis) Figure Q2.9: A-EYE-Web: Diagnosis and Recommendation (a set
    of differential diagnoses) Q1.Table 1 : An employee’s daily query record on EyeCoach
    chatbot. Work pattern A: The daily query record of a software engineer. Day index
    Total number of Type of query Type of query Type of query queries (ergonomics)
    (visual habits) (eye exercises) [REDACTED_PHONE] … … … … … [REDACTED_PHONE] …
    … … … … Work pattern B: The daily query record of a sales executive. [REDACTED_PHONE]
    … … … … … [REDACTED_PHONE] … … … … … Entity Examples Environmental stimuli Perceive
    function Raw data Recognize function Information Learn function Knowledge (new/known)
    Reason function Plan/Solution Act function Effects of action Variable name Values
    Description Patient Name String Patient NRIC String Unique identifier for the
    patient (the patients NRIC) Registration Date Date Date that the patient first
    registered with the doctor Gender M, F The gender of the patient, either Male
    or Female Date of Birth Date Occupation Category One of 10 categories, e.g. unemployed,
    retired, manual work, office work, computer work, outdoor work etc. Variable name
    Values Description Case ID string Unique identifier for the case Patient ID string
    Unique identifier for the patient Diagnosis Date Date Date the diagnosis was performed
    Blood Pressure True, False Is the patient currently taking medication for high
    blood pressure? Diabetes True, False Is the patient currently taking medication
    for diabetes? Cholesterol True, False Is the patient currently taking medication
    for high cholesterol? Other Meds True, False Is the patient currently taking any
    other medication? Symptom1 True, False Is symptom1 present? Symptom2 True, False
    Is symptom2 present? … … …. Symptom40 True, False Is symptom40 present? (extended
    beyond 11 primary and 7 secondary symptoms) Sign1 True, False Is sign1 present?
    Sign2 True, False Is sign2 present? … Sign10 True, False Is sign10 present? (extended
    beyond 3 signs) System Diagnosis List of categories A list of up to 10 diagnoses
    generated by the A-Eye-Web system. Doctor Diagnosis Category The diagnosis given
    by the doctor Treatments List of categories A list of treatments prescribed for
    the patient. This will be blank if no treatments were prescribed Patient Feedback
    Integer from 1 to 5 A number given by the patient indicating how effective they
    thought the treatment was (1 = very ineffective, 5 = very effective). This field
    is blank if no feedback was given.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Dear Participant Greetings from NUS -ISS! We are honoured to partner you in your lifelong learning journey, as you climb the next curve of digital excellence. Our aim is to provide you with the necessary skills and knowledge to aid in your professional and personal growth. Please be advised that the course materials, including the course notes, are protected by copyright and are property of the National University of Singapore or t heir respective copyright owners. The course notes provided are for personal use and reference during the course delivery, and any broadcasting, dissemination, distribution, preparation, reproduction, sub- licensing, transmission, uploading, or of derivativ e works is strictly prohibited. NUS -ISS is dedicated to supporting your growth through a comprehensive portfolio of multiple learning pathways, with a wide spectrum of programmes in critical industry disciplines, including artificial intelligence, cybersecurity, data science, digital governance, digital innovation, smart health, software development and more. Should you require any assistance or information regarding our blended learning programmes, executive programmes, graduate and stackable programmes , please do not hesitate to reach out to us at ask [REDACTED_EMAIL] or via phone at ([REDACTED_PHONE]. Wishing you a productive and fulfilling learning experience. Kind Regards, Yum Hui Yuen (Mrs) Director Executive Education & Corporate Services NUS -ISS National University of Singapore',
    'Dear Participant Greetings from NUS -ISS! We are honoured to partner you in your lifelong learning journey, as you climb the next curve of digital excellence. Our aim is to provide you with the necessary skills and knowledge to aid in your professional and personal growth. Please be advised that the course materials, including the course notes, are protected by copyright and are property of the National University of Singapore or t heir respective copyright owners. The course notes provided are for personal use and reference during the course delivery, and any broadcasting, dissemination, distribution, preparation, reproduction, sub- licensing, transmission, uploading, or of derivativ e works is strictly prohibited. NUS -ISS is dedicated to supporting your growth through a comprehensive portfolio of multiple learning pathways, with a wide spectrum of programmes in critical industry disciplines, including artificial intelligence, cybersecurity, data science, digital governance, digital innovation, smart health, software development and more. Should you require any assistance or information regarding our blended learning programmes, executive programmes, graduate and stackable programmes , please do not hesitate to reach out to us at ask [REDACTED_EMAIL] or via phone at ([REDACTED_PHONE]. Wishing you a productive and fulfilling learning experience. Kind Regards, Yum Hui Yuen (Mrs) Director Executive Education & Corporate Services NUS -ISS National University of Singapore',
    'Master of Technology in Intelligent Systems page 14 of 25 MTech IS Grad Cert Exam: Intelligent Reasoning Systems Figure Q2.8: A-EYE-Web: Diagnosis and Recommendation (single diagnosis) Figure Q2.9: A-EYE-Web: Diagnosis and Recommendation (a set of differential diagnoses) Q1.Table 1 : An employee’s daily query record on EyeCoach chatbot. Work pattern A: The daily query record of a software engineer. Day index Total number of Type of query Type of query Type of query queries (ergonomics) (visual habits) (eye exercises) [REDACTED_PHONE] … … … … … [REDACTED_PHONE] … … … … … Work pattern B: The daily query record of a sales executive. [REDACTED_PHONE] … … … … … [REDACTED_PHONE] … … … … … Entity Examples Environmental stimuli Perceive function Raw data Recognize function Information Learn function Knowledge (new/known) Reason function Plan/Solution Act function Effects of action Variable name Values Description Patient Name String Patient NRIC String Unique identifier for the patient (the patients NRIC) Registration Date Date Date that the patient first registered with the doctor Gender M, F The gender of the patient, either Male or Female Date of Birth Date Occupation Category One of 10 categories, e.g. unemployed, retired, manual work, office work, computer work, outdoor work etc. Variable name Values Description Case ID string Unique identifier for the case Patient ID string Unique identifier for the patient Diagnosis Date Date Date the diagnosis was performed Blood Pressure True, False Is the patient currently taking medication for high blood pressure? Diabetes True, False Is the patient currently taking medication for diabetes? Cholesterol True, False Is the patient currently taking medication for high cholesterol? Other Meds True, False Is the patient currently taking any other medication? Symptom1 True, False Is symptom1 present? Symptom2 True, False Is symptom2 present? … … …. Symptom40 True, False Is symptom40 present? (extended beyond 11 primary and 7 secondary symptoms) Sign1 True, False Is sign1 present? Sign2 True, False Is sign2 present? … Sign10 True, False Is sign10 present? (extended beyond 3 signs) System Diagnosis List of categories A list of up to 10 diagnoses generated by the A-Eye-Web system. Doctor Diagnosis Category The diagnosis given by the doctor Treatments List of categories A list of treatments prescribed for the patient. This will be blank if no treatments were prescribed Patient Feedback Integer from 1 to 5 A number given by the patient indicating how effective they thought the treatment was (1 = very ineffective, 5 = very effective). This field is blank if no feedback was given.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 4,747 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             |
  | details | <ul><li>min: 3 tokens</li><li>mean: 199.2 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 199.2 tokens</li><li>max: 256 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>S-RTAVS/Video analytics foundation/V3.1 © 2025 National University of Singapore. All Rights Reserved Page 50 Practical issue of object tracking • How toinitialize theobject inthefirstframe toactivate thetracker? •Manual initialization (different users have different targets) •Object detector (alistofpre-defined objects) • Where tofindlabelled data fortraining aobject tracker? •Unfortunately, there isnoImageNet dataset inobject tracking research domain. •Fortunately, wedon’t need tore-train thetracker, since thetracker should begeneral tohandle various types ofobjects .The trackers willhandle object tracking inanonline mode. • Speed isextremely important requirement forreal-time tracker . Le ft i ma ge R igh t im ag e Each image has a size of pixels, each block has a size 10 × of pixels (row coordinate, column coordinate) Motion vector 8 Block center in current frame 3 × 3 Either method is okay. • Current – previous: • Previous – current: (0,1) Best matched block center in previous fram...</code> | <code>S-RTAVS/Video analytics foundation/V3.1 © 2025 National University of Singapore. All Rights Reserved Page 50 Practical issue of object tracking • How toinitialize theobject inthefirstframe toactivate thetracker? •Manual initialization (different users have different targets) •Object detector (alistofpre-defined objects) • Where tofindlabelled data fortraining aobject tracker? •Unfortunately, there isnoImageNet dataset inobject tracking research domain. •Fortunately, wedon’t need tore-train thetracker, since thetracker should begeneral tohandle various types ofobjects .The trackers willhandle object tracking inanonline mode. • Speed isextremely important requirement forreal-time tracker . Le ft i ma ge R igh t im ag e Each image has a size of pixels, each block has a size 10 × of pixels (row coordinate, column coordinate) Motion vector 8 Block center in current frame 3 × 3 Either method is okay. • Current – previous: • Previous – current: (0,1) Best matched block center in previous fram...</code> |
  | <code>S-VSE/Vision systems foundation/V3.4 © 2025 National University of Singapore. All Rights Reserved Page 13 Classification: Is it an indoor scene? • 3D from multi-view and sensors • 3D from single images • Adversarial attack and defense • Autonomous driving • Biometrics • Computational imaging • Computer vision for social good • Computer vision theory • Datasets and evaluation • Deep learning architectures and techniques • Document analysis and understanding • Efficient and scalable vision • Embodied vision: Active agents, simulation • Explainable computer vision • Humans: Face, body, pose, gesture, movement • Image and video synthesis and generation • Low-level vision • Machine learning (other than deep learning) • Medical and biological vision, cell microscopy • Multimodal learning • Optimization methods (other than deep learning) • Photogrammetry and remote sensing • Physics-based vision and shape-from-X • Recognition: Categorization, detection, retrieval • Representation learning • R...</code> | <code>S-VSE/Vision systems foundation/V3.4 © 2025 National University of Singapore. All Rights Reserved Page 13 Classification: Is it an indoor scene? • 3D from multi-view and sensors • 3D from single images • Adversarial attack and defense • Autonomous driving • Biometrics • Computational imaging • Computer vision for social good • Computer vision theory • Datasets and evaluation • Deep learning architectures and techniques • Document analysis and understanding • Efficient and scalable vision • Embodied vision: Active agents, simulation • Explainable computer vision • Humans: Face, body, pose, gesture, movement • Image and video synthesis and generation • Low-level vision • Machine learning (other than deep learning) • Medical and biological vision, cell microscopy • Multimodal learning • Optimization methods (other than deep learning) • Photogrammetry and remote sensing • Physics-based vision and shape-from-X • Recognition: Categorization, detection, retrieval • Representation learning • R...</code> |
  | <code>ATA\ S-PRMLS \Day2a.ppt \V3.1 © 2024 National University of Singapore. All Rights Reserved 28 Simple SOM (Kohonen) Network —One-dimension — •A simple 1 -dimensional Kohonen (SOM) network •Input pattern vector fully connected to all neurons •Lateral interactions between neurons constrain activations to spatially bounded ‘excitation zone’ INPUTSKOHONEN LAYERCOMPETITIVE ONE WINNER NEURONS COMPETE AND ONLY ONE "WINS" PATTERN RECOGNITION AND MACHINE LEARNING SYSTEMS DAY 2A Dr Zhu Fangming NUS-ISS [REDACTED_EMAIL] Not be reproduced in any form or by any means, without the written permission of ISS, NUS, other than for the purpose for which it has been supplied. ATA\S-PRMLS\Day2a.ppt\V3.1 © 2024 National University of Singapore. All Rights Reserved [REDACTED_PHONE] Neural Network Models and Designs ATA\S-PRMLS\Day2a.ppt\V3.1 © 2024 National University of Singapore. All Rights Reserved 2 Topics • Radial Basis Function Networks • General Regression Neural Networks • Self-Organizing Map (SOM, Ko...</code> | <code>ATA\ S-PRMLS \Day2a.ppt \V3.1 © 2024 National University of Singapore. All Rights Reserved 28 Simple SOM (Kohonen) Network —One-dimension — •A simple 1 -dimensional Kohonen (SOM) network •Input pattern vector fully connected to all neurons •Lateral interactions between neurons constrain activations to spatially bounded ‘excitation zone’ INPUTSKOHONEN LAYERCOMPETITIVE ONE WINNER NEURONS COMPETE AND ONLY ONE "WINS" PATTERN RECOGNITION AND MACHINE LEARNING SYSTEMS DAY 2A Dr Zhu Fangming NUS-ISS [REDACTED_EMAIL] Not be reproduced in any form or by any means, without the written permission of ISS, NUS, other than for the purpose for which it has been supplied. ATA\S-PRMLS\Day2a.ppt\V3.1 © 2024 National University of Singapore. All Rights Reserved [REDACTED_PHONE] Neural Network Models and Designs ATA\S-PRMLS\Day2a.ppt\V3.1 © 2024 National University of Singapore. All Rights Reserved 2 Topics • Radial Basis Function Networks • General Regression Neural Networks • Self-Organizing Map (SOM, Ko...</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Framework Versions
- Python: 3.11.0
- Sentence Transformers: 3.4.1
- Transformers: 4.49.0
- PyTorch: 2.6.0+cu126
- Accelerate: 1.6.0
- Datasets: 3.5.0
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->
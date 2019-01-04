from nltk.corpus import semcor, LazyCorpusLoader, SemcorCorpusReader
from nltk.corpus import wordnet
from nltk.corpus.reader import XMLCorpusView, XMLCorpusReader
from nltk.corpus.reader.util import *
from itertools import chain


def split_file(file, sep=",", block_size=16384):
    last_fragment = ""
    while True:
        block = file.read(block_size)
        if not block:
            break
        block_fragments = iter(block.split(sep))
        last_fragment += next(block_fragments)
        for fragment in block_fragments:
            yield last_fragment
            last_fragment = fragment
    if last_fragment != "":
        yield last_fragment


def read_files(files, sep=",", block_size=16384):
    if isinstance(files, str):
        files = [files]

    for f in files:
        with open(f, 'r') as ff:
            yield from split_file(ff, sep=sep, block_size=block_size)


class SemCorReader(XMLCorpusReader):

    def __init__(self, root, fileids, wrap_etree=False):
        super(SemCorReader, self).__init__(root, fileids, wrap_etree)

    def instances(self, name='semcor+omsti', fileid=None):
        """

        :param name:
        :param fileid:
        :return: example:
        [
        (id, token),
        (id, token)
        ]
        """
        return self._words(name=name, unit="instance", fileid=fileid)

    def sentences(self, name='semcor+omsti', fileid=None):
        """

        :param name:
        :param fileid:
        :return: example:
        [
        (lemma, pos, token),
        (lemma, pos, token),
        ]
        """
        return self._words(name=name, unit="sentence", fileid=fileid)

    def both(self, name='semcor+omsti', fileid=None):
        """

        :param name:
        :param fileid:
        :return: example:
        equals to (self.instances(), self.sentences())
        """
        return self._words(name=name, unit="both", fileid=fileid)

    def golds(self, name='semcor+omsti', fileid=None):
        if isinstance(name, str):
            name = [name]

        if not fileid:
            fileid = self.fileid(name, suffex=".gold.key.txt")

        file_paths = self.abspaths(fileid)
        sentence_key = ''
        records = []

        # group record by sentence
        for record in read_files(file_paths, sep='\n'):
            record = record.split(' ')
            key = record[0]
            gold = record[1:]
            new_key = key.split('.')[1]
            if new_key != sentence_key:
                if len(records) != 0:
                    yield records
                records = []
                sentence_key = new_key
            records.append((key, gold))
        yield records

    def _find_id(self, name, suffex=".data.xml"):
        end = name + suffex
        return next(x for x in self.fileids() if x.endswith(end.lower()))

    def fileid(self, name, suffex=".data.xml"):
        if isinstance(name, str):
            return [self._find_id(name, suffex)]

        return [self._find_id(x, suffex) for x in name]

    def _words(self,name='semcor+omsti', unit="both", fileid=None):
        """

        :param name:
        :param unit:    "sentence" | "instance" | "both", default: "both".
        :param fileid:
        :return:
        """


        if not fileid:
            fileid = self.fileid(name, suffex=".data.xml")

        abspath = self.abspaths(fileid)
        return concat([
            SemCorView(x, unit=unit) for x in abspath
        ])


class SemCorView(XMLCorpusView):

    def __init__(self, fileid, unit='both', elt_handler=None):
        """

        :param fileid:
        :param unit: "sentence" | "instance" | "both", default "both".
        :param elt_handler:
        """

        tagspec = '.*/sentence'

        self._unit=unit

        super(SemCorView, self).__init__(fileid,tagspec,elt_handler)

    def handle_elt(self, elt, context):
        return self.handle_sentence(elt)

    def handle_sentence(self, elt):

        if self._unit == 'both':
            sent = []
            inst = []
            for child in elt:
                cc = self.handle_word(child)
                if child.tag == 'instance':        # id, lemma, pos, tkn
                    inst.append((cc[:3]))      # id, lemma, pos
                    sent.append(cc[1:])             # lemma, pos, tkn
                else:
                    sent.append(cc)                 # lemma, pos, tkn

            return inst, sent

        return [
            self.handle_word(child) for child in elt
        ]

    def handle_word(self, elt):
        tkn = elt.text
        if not tkn:
            tkn = ""

        lemma = elt.get('lemma', tkn)
        pos = elt.get('pos')

        if elt.tag == 'instance':
            id = elt.get('id')
            if self._unit in ['instance', 'both']:
                return id, lemma, pos, tkn

        return lemma, pos, tkn


new_semcor =  LazyCorpusLoader(
    'new_semcor', SemCorReader, r'(.*\.xml)|(.*\.gold\.key\.txt)', wordnet
)

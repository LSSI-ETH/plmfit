import os

from .util import as_biopython_seq

class GenomeReader:

    '''
    An API to read sequences from the reference genome, assuming it's organized into per-chromosome file.
    A GenomeReader object is initialized from a directory containing the relevant FAST files. It will automatically detect all .fa and .fasta files
    within that directory.
    The reference genome sequences of all human chromosomes (chrXXX.fa.gz files) can be downloaded from UCSC's FTP site at:
    ftp://hgdownload.cse.ucsc.edu/goldenPath/hg19/chromosomes/ (for version hg19)
    ftp://hgdownload.cse.ucsc.edu/goldenPath/hg38/chromosomes/ (for version hg38/GRCh38)
    The chrXXX.fa.gz files need to be uncompressed to obtain chrXXX.fa files.
    IMPORTANT: In version hg19 there's an inconsistency in the reference genome of the M chromosome between UCSC and RegSeq/GENCODE,
    so the file chrM.fa is better to be taken from RefSeq (NC_012920.1) instead of UCSC, from:
    https://www.ncbi.nlm.nih.gov/sviewer/viewer.cgi?tool=portal&save=file&log$=seqview&db=nuccore&report=fasta&sort=&id=251831106&from=begin&to=end&maxplex=1
    In GRCh38 all UCSC files are fine.
    '''
    
    def __init__(self, ref_genome_dir):
        self.ref_genome_dir = ref_genome_dir
        self.chromosome_readers = dict(self._create_chromosome_reader(file_name) for file_name in os.listdir(self.ref_genome_dir) \
                if _is_ref_genome_file(file_name))

    def read_seq(self, chromosome, start, end):
        try:
            return self.chromosome_readers[_find_chrom(chromosome, self.chromosome_readers.keys())].read_seq(start, end)
        except KeyError:
            raise ValueError('Chromosome "%s" was not found in the reference genome.' % chromosome)
        
    def close(self):
        for chromosome_reader in self.chromosome_readers.values():
            chromosome_reader.file_handler.close()
            
    def __contains__(self, chromosome):
        return _find_chrom(chromosome, self.chromosome_readers.keys()) is not None
        
    def _create_chromosome_reader(self, file_name):
        chr_name = file_name.split('.')[0].replace('chr', '')
        f = open(os.path.join(self.ref_genome_dir, file_name), 'r')
        return chr_name, ChromosomeReader(f)

class ChromosomeReader:

    def __init__(self, file_handler):
        self.file_handler = file_handler
        self.header_len = len(file_handler.readline())
        self.line_len = len(file_handler.readline()) - 1

    def read_seq(self, start, end):

        absolute_start = self.convert_to_absolute_coordinate(start)
        absolute_length = self.convert_to_absolute_coordinate(end) - absolute_start + 1

        self.file_handler.seek(absolute_start)
        seq = self.file_handler.read(absolute_length).replace('\n', '').upper()
        return as_biopython_seq(seq)
        
    def convert_to_absolute_coordinate(self, position):
        position_zero_index = position - 1
        return self.header_len + position_zero_index + (position_zero_index // self.line_len)
    
def _find_chrom(query_chr_name, available_chr_names):

    assert isinstance(query_chr_name, str), 'Unexpected chromosome type: %s' % type(query_chr_name)

    if query_chr_name.lower().startswith('chr'):
        query_chr_name = query_chr_name[3:]
        
    query_chr_name = query_chr_name.upper()
    
    for possible_chr_name in _find_synonymous_chr_names(query_chr_name):
        
        if possible_chr_name in available_chr_names:
            return possible_chr_name
            
        prefixed_possible_chr_name = 'chr%s' % possible_chr_name
            
        if prefixed_possible_chr_name in available_chr_names:
            return prefixed_possible_chr_name
            
    return None
    
def _is_ref_genome_file(file_name):
    
    file_name = file_name.lower()
    
    for extension in _SUPPORTED_REF_GENOME_EXTENSIONS:
        if file_name.endswith(file_name):
            return True
            
    return False

def _find_synonymous_chr_names(chr_name):
    
    for synonymous_chr_name_group in _SYNONYMOUS_CHR_NAME_GROUPS:
        if chr_name in synonymous_chr_name_group:
            return synonymous_chr_name_group
            
    # Single-digit numbers can either appear with or without a trailing 0.
    if chr_name.isdigit() and len(str(int(chr_name))) == 1:
        chr_number = str(int(chr_name))
        return {chr_number, '0' + chr_number}
            
    return {chr_name}
    
_SUPPORTED_REF_GENOME_EXTENSIONS = [
    '.fa',
    '.fasta',
]

_SYNONYMOUS_CHR_NAME_GROUPS = [
    {'X', '23'},
    {'Y', '24'},
    {'XY', '25'},
    {'M', 'MT', '26'},
]

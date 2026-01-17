#!/usr/bin/env python3
"""
Script to filter Pfam-A.fasta.gz to only include sequences from specified PFAM families.
"""

import gzip
import os

# Define the PFAM families to keep
TARGET_PFAMS = {
'PF00005',
'PF00006',
'PF00009',
'PF00025',
'PF00063',
'PF00071',
'PF00142',
'PF00154',
'PF00158',
'PF00176',
'PF00225',
'PF00265',
'PF00270',
'PF00271',
'PF00308',
'PF00350',
'PF00406',
'PF00437',
'PF00448',
'PF00485',
'PF00488',
'PF00493',
'PF00503',
'PF00519',
'PF00580',
'PF00625',
'PF00685',
'PF00693',
'PF00709',
'PF00735',
'PF00910',
'PF00931',
'PF01043',
'PF01057',
'PF01078',
'PF01121',
'PF01202',
'PF01268',
'PF01443',
'PF01580',
'PF01583',
'PF01591',
'PF01637',
'PF01656',
'PF01695',
'PF01712',
'PF01715',
'PF01745',
'PF01926',
'PF01935',
'PF02223',
'PF02224',
'PF02263',
'PF02283',
'PF02367',
'PF02374',
'PF02399',
'PF02421',
'PF02456',
'PF02463',
'PF02492',
'PF02499',
'PF02500',
'PF02534',
'PF02562',
'PF02572',
'PF02606',
'PF02689',
'PF02702',
'PF02841',
'PF03028',
'PF03029',
'PF03192',
'PF03193',
'PF03205',
'PF03215',
'PF03237',
'PF03266',
'PF03308',
'PF03354',
'PF03567',
'PF03618',
'PF03668',
'PF03796',
'PF03846',
'PF03969',
'PF03976',
'PF04084',
'PF04257',
'PF04275',
'PF04310',
'PF04317',
'PF04466',
'PF04548',
'PF04665',
'PF04670',
'PF04851',
'PF05049',
'PF05127',
'PF05179',
'PF05272',
'PF05496',
'PF05609',
'PF05621',
'PF05625',
'PF05673',
'PF05707',
'PF05729',
'PF05763',
'PF05783',
'PF05872',
'PF05876',
'PF05879',
'PF05894',
'PF05970',
'PF06048',
'PF06068',
'PF06144',
'PF06309',
'PF06414',
'PF06418',
'PF06431',
'PF06564',
'PF06733',
'PF06745',
'PF06858',
'PF06911',
'PF06990',
'PF07015',
'PF07034',
'PF07088',
'PF07517',
'PF07652',
'PF07693',
'PF07724',
'PF07726',
'PF07728',
'PF07755',
'PF07931',
'PF07999',
'PF08245',
'PF08298',
'PF08303',
'PF08351',
'PF08423',
'PF08433',
'PF08438',
'PF08477',
'PF08903',
'PF09037',
'PF09140',
'PF09439',
'PF09547',
'PF09711',
'PF09807',
'PF09818',
'PF09820',
'PF09848',
'PF10088',
'PF10236',
'PF10412',
'PF10443',
'PF10483',
'PF10609',
'PF10649',
'PF10662',
'PF10923',
'PF10996',
'PF11111',
'PF11398',
'PF11496',
'PF11602',
'PF12128',
'PF12344',
'PF12399',
'PF12696',
'PF12774',
'PF12775',
'PF12780',
'PF12781',
'PF12846',
'PF12848',
'PF13086',
'PF13087',
'PF13166',
'PF13173',
'PF13175',
'PF13177',
'PF13189',
'PF13191',
'PF13207',
'PF13238',
'PF13245',
'PF13304',
'PF13307',
'PF13337',
'PF13361',
'PF13401',
'PF13469',
'PF13476',
'PF13479',
'PF13481',
'PF13500',
'PF13514',
'PF13521',
'PF13538',
'PF13555',
'PF13558',
'PF13604',
'PF13614',
'PF13654',
'PF13671',
'PF13871',
'PF13872',
'PF14396',
'PF14417',
'PF14516',
'PF14532',
'PF14617',
'PF16203',
'PF16260',
'PF16575',
'PF16796',
'PF16813',
'PF16834',
'PF16836',
'PF17213',
'PF17784',
'PF18082',
'PF18128',
'PF18133',
'PF18747',
'PF18748',
'PF18751',
'PF18766',
'PF19044',
'PF19263',
'PF19518',
'PF19557',
'PF19568',
'PF19798',
'PF19842',
'PF19975',
'PF19993',
'PF19995',
'PF20030',
'PF20307',
'PF20454',
'PF20692',
'PF20693',
'PF20702',
'PF20703',
'PF20720',
'PF21090',
'PF21228',
'PF21264',
'PF21445',
'PF21448',
'PF21449',
'PF22232',
'PF22298',
'PF22527',
'PF22590',
'PF22679',
'PF22916',
'PF23365',
'PF23415',
'PF23442',
'PF23569',
'PF23867',
'PF24179',
'PF24336',
'PF24389',
'PF24404',
'PF24406',
'PF24786',
'PF24883',
'PF25199',
'PF25201',
'PF25496',
'PF25683'
}

# Input and output paths
INPUT_FILE = 'data/pfam/Pfam-A.fasta.gz'
OUTPUT_FILE = 'data/pfam/Pfam-A-filtered.fasta.gz'


def filter_pfam_fasta(input_path: str, output_path: str, target_pfams: set):
    """
    Filter a Pfam FASTA file to only include sequences from specified families.
    
    Args:
        input_path: Path to input .gz FASTA file
        output_path: Path to output .gz FASTA file
        target_pfams: Set of PFAM family IDs to keep
    """
    sequences_written = 0
    families_found = set()
    current_family = None
    write_current = False
    
    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")
    print(f"Target families: {sorted(target_pfams)}")
    print()
    
    with gzip.open(input_path, 'rt') as f_in, gzip.open(output_path, 'wt') as f_out:
        for line in f_in:
            if line.startswith('>'):
                # Parse the PFAM family from the header
                # Format: >SEQUENCE_ID DESCRIPTION PF00005.XX;
                # Family is extracted as: line.split()[-1].split(';')[0]
                family_with_version = line.split()[-1].split(';')[0]
                # Extract just the family ID without version (e.g., PF00005.XX -> PF00005)
                family = family_with_version.split('.')[0]
                current_family = family
                
                if family in target_pfams:
                    write_current = True
                    families_found.add(family)
                    f_out.write(line)
                    sequences_written += 1
                else:
                    write_current = False
            else:
                # This is a sequence line
                if write_current:
                    f_out.write(line)
    
    print(f"Filtering complete!")
    print(f"Total sequences written: {sequences_written}")
    print(f"Families found and included: {sorted(families_found)}")
    
    # Check for missing families
    missing = target_pfams - families_found
    if missing:
        print(f"WARNING: The following families were NOT found in the input file: {sorted(missing)}")
    
    return sequences_written, families_found


if __name__ == '__main__':
    # Get the base directory (script location)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_path = os.path.join(base_dir, INPUT_FILE)
    output_path = os.path.join(base_dir, OUTPUT_FILE)
    
    # Verify input file exists
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Run the filtering
    filter_pfam_fasta(input_path, output_path, TARGET_PFAMS)

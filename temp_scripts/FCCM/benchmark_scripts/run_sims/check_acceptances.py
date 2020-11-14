import argparse

def compute_overacceptance_counts(base_file, split_acceptance_files):
    # First, read the base acceptance indexes.
    with open(base_file, 'r') as f:
        accept_indexes = set()
        for line in f.readlines():
            if line == '':
                continue

            accept_index = line.split(',')[1].replace(')', '').replace(' ', '')
            start_index = line.split(',')[0].replace('(', '')
            accept_indexes.add((int(start_index), int(accept_index)))

        # Now, compute the other ones.  Assume that the last file
        # contains the overall acceptance information.
        # Since it seems that we are currently only generating
        # one file for things, this shouldn't matter too much.
        # However, I do note that this should support a whole
        # micrograph structure.
        acceptance_file_acceptances = set()
        for file in split_acceptance_files:
            print file
            with open(file, 'r') as f:
                for line in f.readlines():
                    if line == '':
                        continue

                    accept_index = line.split(',')[1].replace(')', '').replace(' ', '')
                    acceptance_file_acceptances.add(int(accept_index))

        for (start, end) in accept_indexes:
            # Check that there is at least one accept in this range --- i.e. we did not miss an acceptance we should have had.
            found_match = False
            found_end = None
            for accept_index in acceptance_file_acceptances:
                if accept_index >= start and accept_index <= end:
                    found_match = True
                    found_end = end
            if not found_match:
                print "Failed to find a match for pattern the underlying accelerator managed!"
            else:
                print "Found correct prefix match (missing ", found_end - accept_index, " bytes from full match"

        # Now, check how many things are overapproximations:
        for accept_index in acceptance_file_acceptances:
            for (start, end) in accept_indexes:
                found_match = False
                if accept_index >= start or accept_index <= end:
                    found_match = True

            if not found_match:
                print "Found an overacceptance!"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ground_truth_file')
    parser.add_argument('split_acceptance_files', nargs='+')

    args = parser.parse_args()

    compute_overacceptance_counts(args.ground_truth_file, args.split_acceptance_files)

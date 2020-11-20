import argparse

def compute_overacceptance_counts(base_file, split_acceptance_files):
    failed_matches = 0
    correct_matches = 0
    overacceptances = 0

    # First, read the base acceptance indexes.
    with open(base_file, 'r') as f:
        accept_indexes = set()
        average_accept_lengths = []
        for line in f.readlines():
            if line == '':
                continue

            accept_index = int(line.split(',')[1].replace(')', '').replace(' ', ''))
            start_index = int(line.split(',')[0].replace('(', ''))
            accept_indexes.add((start_index, accept_index))
            average_accept_lengths.append(accept_index - start_index)
        # This allows us to estimate the number of bytes that we must send to the CPU.
        if len(average_accept_lengths) > 0:
            max_accept_length = max(average_accept_lengths)
            average_accept_lengths = sum(average_accept_lengths) / len(average_accept_lengths)
        else:
            max_accept_length = 0
            average_accept_lengths = 0

        # Now, compute the other ones.  Assume that the last file
        # contains the overall acceptance information.
        # Since it seems that we are currently only generating
        # one file for things, this shouldn't matter too much.
        # However, I do note that this should support a whole
        # micrograph structure.
        acceptance_file_acceptances = set()
        all_files_accepts = []
        for file in split_acceptance_files:
            print file
            with open(file, 'r') as f:
                this_file_accepts = {}
                for line in f.readlines():
                    if line == '':
                        continue

                    start_index = int(line.split(',')[0].replace('(', '').strip())
                    accept_index = int(line.split(',')[1].replace(')', '').replace(' ', '').strip())
                    if start_index in this_file_accepts:
                        this_file_accepts[start_index].append(accept_index)
                    else:
                        this_file_accepts[start_index] = [accept_index]
                all_files_accepts.append(this_file_accepts)


        # Now, combine all those into one.
        # You can hop to any subsequent sims.  If you don't hop through
        # all the sims, then you aren't an accept!
        acceptance_file_acceptances = set()
        initial_accepts = all_files_accepts[0]
        for i in range(1, len(all_files_accepts)):
            this_accepts_set = all_files_accepts[i]
            current_accepts = set()
            for start in initial_accepts:
                ends = initial_accepts[start]
                for end in ends:
                    if end in this_accepts_set:
                        for ind in this_accepts_set[end]:
                            current_accepts.add(ind)
            initial_accepts = current_accepts
        acceptance_file_acceptances = initial_accepts

        for (start, end) in accept_indexes:
            # Check that there is at least one accept in this range --- i.e. we did not miss an acceptance we should have had.
            found_match = False
            found_end = None
            for accept_index in acceptance_file_acceptances:
                if accept_index >= start and accept_index <= end:
                    found_match = True
                    found_end = end
            if not found_match:
                # print "Failed to find a match for pattern the underlying accelerator managed!"
                failed_matches += 1
            else:
                correct_matches += 1
                # print "Found correct prefix match (missing ", found_end - accept_index, " bytes from full match"

        # Now, check how many things are overapproximations:
        for accept_index in acceptance_file_acceptances:
            found_match = False
            for (start, end) in accept_indexes:
                if accept_index >= start and accept_index <= end:
                    found_match = True

            if not found_match:
                overacceptances += 1
                # print "Found an overacceptance!"
    print "Evaluated Accelerator output ", base_file
    print "Overacceptances", overacceptances
    print "Successful matches", correct_matches
    print "Failed matches (i.e. a bug)", failed_matches
    print "Average Accept Length", average_accept_lengths
    print "Max Accept Length", max_accept_length

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ground_truth_file')
    parser.add_argument('split_acceptance_files', nargs='+')

    args = parser.parse_args()

    compute_overacceptance_counts(args.ground_truth_file, args.split_acceptance_files)

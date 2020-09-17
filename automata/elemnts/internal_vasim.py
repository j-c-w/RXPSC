def parseSymbolSet(symbol_set):
    bitset = set()
    if symbol_set == "*":
        for i in range(0, 256):
            bitset.add(i)
        return bitset

    # KAA found that apcompile parses symbol-set="." to mean "^\x0a"
    # hard-coding this here
    if symbol_set == ".":
        for i in range(0, 256):
            if i == ord('\n'):
                continue
            bitset.add(i)
        return bitset

    in_charset = False;
    escaped = False;
    inverting = False;
    range_set = False;
    bracket_sem = 0;
    brace_sem = 0;
    last_char = 0;
    range_start = 0;

    # SPECIAL CHAR CODES
    OPEN_BRACKET = 256;

    # handle symbol sets that start and end with curly braces {###}
    if((symbol_set[0] == '{') and
            (symbol_set[symbol_set.size() - 1] == '}')):

        print "CURLY BRACES NOT IMPLEMENTED"
        raise Error()

    index = 0;
    while(index < len(symbol_set)):

        c = symbol_set[index];

        if c == '[':
            if(escaped):
                bitset.add(c);
                if(range_set):
                    setRange(bitset,range_start,c);
                    range_set = False;
                last_char = c;
                escaped = False;
            else:
                last_char = OPEN_BRACKET;
                bracket_sem+=1;
        elif c == ']':
            if(escaped):
                bitset.add(c);
                if(range_set):
                    setRange(bitset,range_start,c);
                    range_set = False;
                escaped = False;
                last_char = c;
            else:
                bracket_sem-=1;
        elif c == '{' :
            bitset.add(c);
            if(range_set):
                setRange(bitset,range_start,c);
                range_set = False;

            last_char = c;
        elif c == '}' :
                bitset.add(c);
                if(range_set):
                    setRange(bitset,range_start,c);
                    range_set = False;
                last_char = c;

            # escape
        elif c == '\\' :
            if(escaped):
                bitset.add(c);
                if(range_set):
                    setRange(bitset,range_start,c);
                    range_set = False;

                last_char = c;
                escaped = False;
            else:
                escaped = True;

            #  escaped chars
        elif c == 'n' :
            if(escaped):
                bitset.add('\n');
                if(range_set):
                    setRange(bitset,range_start,'\n');
                    range_set = False;
                last_char = '\n';
                escaped = False;
            else:
                bitset.add(c);
                if(range_set):
                    setRange(bitset,range_start,c);
                    range_set = False;
                last_char = c;
        elif c == 'r' :
            if(escaped):
                bitset.add('\r');
                if(range_set):
                    setRange(bitset,range_start,'\r');
                    range_set = False;
                last_char = '\r';
                escaped = False;
            else:
                bitset.add(c);
                if(range_set):
                    setRange(bitset,range_start,'\r');
                    range_set = False;
                last_char = c;
        elif c == 't' :
            if(escaped):
                bitset.add('\t');
                if(range_set):
                    setRange(bitset,range_start,'\r');
                    range_set = False;
                last_char = '\t';
                escaped = False;
            else:
                bitset.add(c);
                if(range_set):
                    setRange(bitset,range_start,c);
                    range_set = False;
                last_char = c;
        elif c == 'a' :
            if(escaped):
                bitset.add('\a');
                if(range_set):
                    setRange(bitset,range_start,'\a');
                    range_set = False;
                last_char = '\a';
                escaped = False;
            else:
                bitset.add(c);
                if(range_set):
                    setRange(bitset,range_start,c);
                    range_set = False;
                last_char = c;
        elif c == 'b' :
            if(escaped):
                bitset.add('\b');
                if(range_set):
                    setRange(bitset,range_start,'\b');
                    range_set = False;
                last_char = '\b';
                escaped = False;
            else:
                bitset.add(c);
                if(range_set):
                    # std::cout << "RANGE SET" << std::endl;
                    setRange(bitset,range_start,c);
                    range_set = False;
                last_char = c;
        elif c == 'f' :
            if(escaped):
                bitset.add('\f');
                if(range_set):
                    setRange(bitset,range_start,'\f');
                    range_set = False;
                last_char = '\f';
                escaped = False;
            else:
                bitset.add(c);
                if(range_set):
                    setRange(bitset,range_start,c);
                    range_set = False;
                last_char = c;
        elif c == 'v' :
            if(escaped):
                bitset.add('\v');
                if(range_set):
                    setRange(bitset,range_start,'\v');
                    range_set = False;
                last_char = '\v';
                escaped = False;
            else:
                bitset.add(c);
                if(range_set):
                    setRange(bitset,range_start,c);
                    range_set = False;
                last_char = c;
        elif c == '\'' :
            if(escaped):
                bitset.add('\'');
                if(range_set):
                    setRange(bitset,range_start,'\'');
                    range_set = False;
                last_char = '\'';
                escaped = False;
            else:
                bitset.add(c);
                if(range_set):
                    setRange(bitset,range_start,c);
                    range_set = False;
                last_char = c;
        elif c == '\"' :
            if(escaped):
                bitset.add('\"');
                if(range_set):
                    setRange(bitset,range_start,'\"');
                    range_set = False;
                last_char = '\"';
                escaped = False;
            else:
                bitset.add(c);
                if(range_set):
                    setRange(bitset,range_start,c);
                    range_set = False;
                last_char = c;
            #  Range
        elif c == '-' :
            #  only set the range if the previous char wasn't a bracket
            if(escaped or last_char == OPEN_BRACKET):
                bitset.add('-');
                if(range_set):
                    setRange(bitset,range_start,'-');
                    range_set = False;
                escaped = False;
                last_char = '-';
            else:
                range_set = True;
                range_start = last_char;

            #  Special Classes
        elif c == 's' :
            if(escaped):
                bitset.add('\n');
                bitset.add('\t');
                bitset.add('\r');
                bitset.add('\x0B'); # vertical tab
                bitset.add('\x0C');
                bitset.add('\x20');
                escaped = False;
            else:
                bitset.add(c);
                if(range_set):
                    setRange(bitset,range_start,c);
                    range_set = False;
                last_char = c;

        elif c == 'd' :
            if(escaped):
                setRange(bitset,48,57);
                escaped = False;
            else:
                bitset.add(c);
                if(range_set):
                    setRange(bitset,range_start,c);
                    range_set = False;
                last_char = c;

        elif c == 'w' :
            if(escaped):
                bitset.add('_'); #  '_'
                setRange(bitset,48,57); #  d
                setRange(bitset,65,90); 
                setRange(bitset,97,122); 
                escaped = False;
            else:
                bitset.add(c);
                if(range_set):
                    setRange(bitset,range_start,c);
                    range_set = False;
                last_char = c;

        elif c == '^' :
            if(escaped):
                bitset.add(c);
                if(range_set):
                    setRange(bitset,range_start,c);
                    range_set = False;
                last_char = c;
                escaped = False;
            else:
                inverting = True;

        elif c == 'x' :
            if(escaped):
                # process hex char
                index += 1;
                hex = [0, 0]
                hex[0] = symbol_set[index];
                hex[1] = symbol_set[index+1];
                number = int(''.join(hex), 16);

                # 
                index  += 1;
                bitset.add(number)
                if(range_set):
                    setRange(bitset,range_start,number);
                    range_set = False;
                last_char = number;
                escaped = False;
            else:
                bitset.add(c);
                if(range_set):
                    setRange(bitset,range_start,c);
                    range_set = False;
                last_char = c;

            #  Other characters
        else:
            if(escaped):
                #  we escaped a char that is not valid so treat it normaly
                escaped = False;
            bitset.add(c);
            if(range_set):
                setRange(bitset,range_start,c);
                range_set = False;
            last_char = c;

        index+= 1
    # end char while loop

    if(inverting):
        invert(bitset)

    if(bracket_sem != 0 or brace_sem != 0):
        print "MALFORMED BRACKETS OR BRACES: " 
        print "brackets: ",  bracket_sem
        raise Error()
    return list(bitset)

def invert(col):
    result = set()
    for i in range(0, 256):
        if i not in col:
            result.add(i)
    return result

def setRange(bitset, start, end):
    for i in range(start, end + 1):
        bitset.add(i);

import java.util.StringTokenizer;

public class MattABCTools {

    static boolean flag;

    static String fixNotationForTunepal(String notation) {
	notation = notation.replace("á", "\\\'a");
	notation = notation.replace("é", "\\\'e");
	notation = notation.replace("í", "\\\'i");
	notation = notation.replace("ó", "\\\'o");
	notation = notation.replace("ú", "\\\'u");
	notation = notation.replace("Á", "\\\'A");
	notation = notation.replace("É", "\\\'E");
	notation = notation.replace("Í", "\\\'I");
	notation = notation.replace("Ó", "\\\'O");
	notation = notation.replace("Ú", "\\\'U");

	if (notation.indexOf("I:linebreak $") != -1) {
	    int tuneStart = skipHeaders(notation);
	    String justTune = notation.substring(tuneStart);
	    justTune = justTune.replace("\r", "");
	    justTune = justTune.replace("\r\n", "");
	    justTune = justTune.replace("\n", "");
	    justTune = justTune.replace("$ ", "$");
	    justTune = justTune.replace("$", "\n");
	    justTune = justTune.replaceAll("[^\\n]w:", "\nw:");

	    notation = notation.substring(0, tuneStart) + justTune;
	}
	return notation;
    }

    static String removeExtraNotation(String key) {
	String ret = key.replaceAll(">", "");

	ret = ret.replaceAll("<", "");
	ret = ret.replaceAll("/", "");
	ret = ret.replace("\\", "");
	ret = ret.replace("(", "");
	ret = ret.replace(")", "");
	ret = ret.replace("/", "");
	ret = ret.replace("-", "");
	ret = ret.replace("!", "");
	// ret = ret.replace("_", "");

	// remove guitar chords
	ret = ret.replaceAll("\\[[^\\]]*\\]", "");

	StringBuffer ret1 = new StringBuffer();
	for (int i = 0 ; i < ret.length() ; i ++) {
	    char cur = ret.charAt(i);
	    if (((cur >= 'A') && (cur <= 'G')) || (cur == 'Z') ||
		((cur >= 'a') && (cur <= 'g')) || (cur == 'z') ||
		(cur == '=') || (cur == '^') || (cur == '_'))
		ret1.append(cur);
	}
	ret = ret1.toString();
	return ret;
    }

    static String removeLongNotes(String key) {
	StringBuffer ret = new StringBuffer();
	char lastChar = '*';

	for (int i = 0 ; i < key.length() ; i ++) {
	    char current = key.charAt(i);
	    if (current != lastChar) {
		ret.append(current);
		lastChar = current;
	    }
	}

	return ret.toString();
    }

    public static int skipHeaders(String tune) {
	int i = 0;
	int inChars = 0;
	boolean inHeader = true;

	while ((i < tune.length()) && (inHeader)) {
	    char c = tune.charAt(i);
	    if (inChars == 1)
		if (((c == ':') && (tune.charAt(i-1) != '|'))
		    || ((c == '%') && (tune.charAt(i-1) == '%')))
		    inHeader = true;
		else {
		    inHeader = false;
		    i -=2;
		}
	    if ((c == '\r') || (c == '\n'))
		inChars = -1;
	    i ++;
	    inChars ++;
	}
	return i;
    }

    public static String expandParts(String notes) {
	StringBuffer retValue = new StringBuffer(notes);
	try {
	    int start = 0;
	    int end = 0;
	    String endToken = ":|";
	    int count = 0;
	    while (true) {
		if (count > 10)
		    throw new ArrayIndexOutOfBoundsException("Too many parts in tune" + notes);
		count ++;
		end = retValue.indexOf(endToken);

		if ((end == -1))
		    break;
		else {
		    int newStart = retValue.lastIndexOf("|:", end);
		    if (newStart != -1)
			start = newStart + 2;
		    if ((retValue.length() > end + 2) &&
			Character.isDigit(retValue.charAt(end + 2))) {
			int numSpecialBars = 1;
			StringBuffer expanded = new StringBuffer();
			int normalPart = retValue.lastIndexOf("|", end);
			if (! Character.isDigit(retValue.charAt(normalPart + 1))) {
			    normalPart = retValue.lastIndexOf("|", normalPart - 1);
			    numSpecialBars ++;
			}
			expanded.append(retValue.substring(start, normalPart));
			expanded.append("|");
			expanded.append(retValue.substring(normalPart + 2, end));
			int secondTime = end;
			while ((numSpecialBars --) > 0)
			    secondTime = retValue.indexOf("|", secondTime + 2);
			expanded.append("|");
			expanded.append(retValue.substring(start, normalPart));
			expanded.append("|");
			expanded.append(retValue.substring(end + 3, secondTime));
			expanded.append("|");
			retValue.replace(start, secondTime, expanded.toString());
		    } else {
			StringBuffer expanded = new StringBuffer();
			expanded.append(retValue.substring(start, end));
			expanded.append("|");
			expanded.append(retValue.substring(start, end));
			retValue.replace(start, end + 2, expanded.toString());
			start = start + expanded.toString().length();
		    }
		}
	    }
	} catch (Exception e) {
	    flag = false;
	    retValue = new StringBuffer(notes);
	}
	return retValue.toString();
    }

    public static String stripBarDivisions(String notes) {
	StringBuffer retValue = new StringBuffer();

	for (int i = 0 ;  i < notes.length(); i ++) {
	    char c  = notes.charAt(i);
	    if ((c != '|') && (c != ':'))
		retValue.append(c);
	}
	return retValue.toString();
    }

    public static String removeTripletMarks(String notes)
    {
	StringBuffer retValue = new StringBuffer();
	boolean inOrnament = false;
	for (int i = 0 ;  i < notes.length(); i ++) {
	    char c  = notes.charAt(i);
	    if ((c == '(') && Character.isDigit(notes.charAt(i+1))) {
		i +=1;
		continue;
	    }
	    retValue.append(c);
	}
	return retValue.toString();
    }

    public static String expandLongNotes(String notes)
    {
	StringBuffer retValue = new StringBuffer();
	boolean inOrnament = false;
	for (int i = 0 ;  i < notes.length(); i ++) {
	    char c  = notes.charAt(i);
	    if (c == '{') {
		inOrnament = true;
		continue;
	    }
	    if (c == '}') {
		inOrnament = false;
		continue;
	    }

	    if ((c != '~') &&
		!inOrnament &&
		(c != ',') &&
		// (c != '=') &&
		// (c != '^') &&
		(c != '\''))
		retValue.append(c);
	}
	for (int i = 1 ;  i < retValue.length(); i ++) {
	    char c  = retValue.charAt(i);
	    char p = retValue.charAt(i -1);
	    // Its a long note
	    if (Character.isDigit(c) && Character.isLetter(p)) {
		String expanded = "";
		int howMany = c - '0';
		for (int j = 0 ; j < howMany; j ++)
		    expanded += p;
		retValue.replace(i - 1, i + 1, expanded);
	    }
	}
	return retValue.toString();
    }

    public static String stripNonNotes(String notes)
    {
	StringBuffer retValue = new StringBuffer();
	notes = stripComments(notes);
	for (int i = 0 ;  i < notes.length(); i ++) {
	    char c  = notes.charAt(i);

	    if (((c >= 'A') && (c <= 'Z')) ||
		((c >= 'a') && (c <= 'z')) ||
		((c >= '1') && (c <= '9')) ||
		(c == '(') ||
		(c == '^') || (c == '=') || (c == '_'))
		retValue.append(c);
	}
	return retValue.toString();
    }

    public static String stripWhiteSpace(String transcription)
    {
	StringBuffer retValue = new StringBuffer();
	int i = 0;
	while (i < transcription.length()) {
	    if ((transcription.charAt(i) != ' ') &&
		(transcription.charAt(i) != '\r') &&
		(transcription.charAt(i) != '\n'))
		retValue.append(transcription.charAt(i));
	    i ++;
	}

	return retValue.toString();
    }

    public static String stripComments(String transcription) {
	StringBuffer retValue = new StringBuffer();

	int i = 0;
	boolean inComment = false;
	while (i < transcription.length()) {
	    if (transcription.charAt(i) == '"')
		if (inComment) {
		    inComment = false;
		    i ++;
		    continue;
		} else
		    inComment = true;
	    if (!inComment)
		retValue.append(transcription.charAt(i));
	    i ++;
	}
	return retValue.toString();
    }

    public static String stripAdvancedABC(String body)
    {
	String ret = body;
	ret = ret.replace("!fermata!", "");
	ret = ret.replace("!trill)!", "");
	ret = ret.replace("!trill(!", "");
	ret = ret.replace("!turn!", "");
	return ret;
    }

    public static String stripAll(String key)
    {
	flag = true;
	key = MattABCTools.stripComments(key);
	key = MattABCTools.stripWhiteSpace(key);
	key = MattABCTools.expandLongNotes(key);
	key = MattABCTools.expandParts(key);
	key = MattABCTools.stripBarDivisions(key);
	key = MattABCTools.removeTripletMarks(key);
	key = MattABCTools.removeExtraNotation(key);
	key = MattABCTools.stripAdvancedABC(key);
	key = key.toUpperCase();
	key = key.replaceAll("^Z*", "");
	key = key.replaceAll("Z*$", "");
	return key;
    }

    public static void main(String[] args)
    {
	// String test = args[0];
	String test = "X: 1\nL: 1/4\nK: Gm\n|z4 BA |: G2B2B2 | d4 BA | G2B2B2 | d4 Bc | B2A2A2 | A3 Bcd |[1 e4 d2 | d4 BA :|[2  e2d2F2 | G4 d2 || |: g4 d2 |g4 d2 | e2 d3 e | B4 Bc | B2A2A2 | A3 Bcd |[1 e4 d2 | d4 ef :|[2 e2 d2 F2 | G4 BA|";
	// test += "[a]A2FA df~f2|dfef dB~B2|A2FA dffe|dBAG FDDB|";
	// test += "A2FA df~f2|afef dB~B2|A2FA dffe|dBAG FDD2||";
	// test += "|:a2~a2 afdf|afef dB~B2|fbba bafa|bfaf feef|";
	// test += "bf~f2 af~f2|afef dB~B2|A2FA dffe|1 dBAF ADD2:|2 dBAF ADDB||";
	// test += "[b]A2FA dfef|df (3efe dB~B2|A2FA defe|dBAG FD~D2|";
	// test += "A2FA df~f2|afgf efdB|(3ABA FA defe|dBAG FD~D2||";
	// test += "|:~a3z afdf|afef dB~B2|fbba babc'|d'c'ba feef|";
	// test += "bf~f2 af~f2|afef efdB|(3ABA FA defe|1 dBAF ADD2:|2 dBAF ADD2||";
	System.out.println(stripAll(test));
    }
}

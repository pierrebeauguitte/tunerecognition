// Copyright 2006-2008 Lionel Gueganton
// This file is part of abc4j.
//
// abc4j is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// abc4j is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with abc4j.  If not, see <http://www.gnu.org/licenses/>.

import java.util.Hashtable;
import java.util.Iterator;
import java.util.Vector;

import abc.notation.Accidental;
import abc.notation.BarLine;
import abc.notation.Decoration;
import abc.notation.Interval;
import abc.notation.KeySignature;
import abc.notation.MultiNote;
import abc.notation.Music;
import abc.notation.Note;
import abc.notation.NoteAbstract;
import abc.notation.RepeatBarLine;
import abc.notation.Tempo;
import abc.notation.Tune;
import abc.notation.Voice;

public class ABCSpeller {

    private static final int SEQUENCE_RESOLUTION = Note.QUARTER;

    public static long[] toPitchClassHistogram(Tune tune) {
	long[] pitchClassHistogram = new long[12];
	try {
	    int lastRepeatOpen = -1;
	    int repeatNumber = 1;
	    boolean inWrongEnding = false;
	    KeySignature tuneKey = null;
	    KeySignature currentKey = null;
	    Hashtable partsKey = new Hashtable();

	    long elapsedTime = 0;
	    NoteAbstract[] graceNotes = null;
	    Music staff = tune.getMusicForAudioRendition();
	    Iterator it = staff.getVoices().iterator();
	    while (it.hasNext()) {
		Voice voice = (Voice) it.next();
		int i = 0;
		while (i < voice.size()) {
		    if (!inWrongEnding) {
			if (voice.elementAt(i) instanceof abc.notation.KeySignature) {
			    tuneKey = (KeySignature)(voice.elementAt(i));
			    currentKey = new KeySignature(tuneKey.getAccidentals());
			}
			else
			    if (voice.elementAt(i) instanceof abc.notation.Note
				&& !((abc.notation.Note)voice.elementAt(i)).isEndingTie()) {

				Note note = (Note)voice.elementAt(i);
				long noteDuration;
				boolean fermata = false;
				Vector decorationNotes = new Vector();
				if (note.hasGeneralGracing() || note.hasDecorations()) {
				    Decoration[] d = note.getDecorations();
				    for (int j = 0; j < d.length; j++) {
					switch (d[j].getType()) {
					case Decoration.FERMATA:
					case Decoration.FERMATA_INVERTED:
					    fermata = true; break;
					case Decoration.LOWERMORDENT:
					case Decoration.UPPERMORDENT:
					case Decoration.DOUBLE_LOWER_MORDANT:
					case Decoration.DOUBLE_UPPER_MORDANT:
					case Decoration.TRILL:
					case Decoration.TURN:
					case Decoration.TURN_INVERTED:
					case Decoration.TURNX:
					case Decoration.TURNX_INVERTED:
					    Note n = new Note(note.getHeight());
					    n.setAccidental(note.getAccidental(currentKey));
					    Note o = new Interval(Interval.SECOND, Interval.MAJOR,
								  Interval.UPWARD)
						.calculateSecondNote(n);
					    Note m = new Interval(Interval.SECOND, Interval.MAJOR,
								  Interval.DOWNWARD)
						.calculateSecondNote(n);
					    o.setAccidental(Accidental.NONE);
					    m.setAccidental(Accidental.NONE);
					    n.setStrictDuration(Note.THIRTY_SECOND);
					    m.setStrictDuration(Note.THIRTY_SECOND);
					    o.setStrictDuration(Note.THIRTY_SECOND);
					    switch (d[j].getType()) {
					    case Decoration.DOUBLE_LOWER_MORDANT:
						decorationNotes.add(n);
						decorationNotes.add(m);
					    case Decoration.LOWERMORDENT:
						decorationNotes.add(n);
						decorationNotes.add(m);
						break;
					    case Decoration.DOUBLE_UPPER_MORDANT:
					    case Decoration.TRILL:
						decorationNotes.add(n);
						decorationNotes.add(o);
					    case Decoration.UPPERMORDENT:
						decorationNotes.add(n);
						decorationNotes.add(o);
						break;
					    case Decoration.TURNX_INVERTED:
					    case Decoration.TURN:
						decorationNotes.add(o);
						decorationNotes.add(n);
						decorationNotes.add(m);
						break;
					    case Decoration.TURNX:
					    case Decoration.TURN_INVERTED:
						decorationNotes.add(m);
						decorationNotes.add(n);
						decorationNotes.add(o);
					    }
					    break;
					}
				    }
				}
				long graceNotesDuration = 0;
				if (note.hasGracingNotes() || (decorationNotes.size() > 0)) {
				    graceNotes = note.getGracingNotes();
				    int divisor = 1;
				    if (note.getStrictDuration() >= Note.HALF)
					divisor = 1; //grace is an eighth
				    else if (note.getStrictDuration() >= Note.QUARTER)
					divisor = 2; //16th
				    else if (note.getStrictDuration() >= Note.EIGHTH)
					divisor = 4; //32nd
				    else
					divisor = 8; //64th
				    if (note.hasGracingNotes()) {
					for (int j=0;j<graceNotes.length;j++) {
					    noteDuration = getNoteLengthInTicks(graceNotes[j], staff)/divisor;
					    graceNotesDuration += noteDuration;
					    if (graceNotes[j] instanceof Note)
						playNote((Note)graceNotes[j], i, currentKey, elapsedTime,
							 noteDuration, pitchClassHistogram);
					    else
						playMultiNote((MultiNote)graceNotes[j], i, currentKey,
							      noteDuration, staff, pitchClassHistogram);
					    elapsedTime+=noteDuration;
					}
				    }
				    for (int j=0;j<decorationNotes.size();j++) {
					noteDuration = getNoteLengthInTicks((Note)decorationNotes.elementAt(j), staff);
					graceNotesDuration += noteDuration;
					playNote((Note)decorationNotes.elementAt(j), i, currentKey, elapsedTime,
						 noteDuration, pitchClassHistogram);
					elapsedTime+=noteDuration;
				    }
				}
				noteDuration = getNoteLengthInTicks(note, staff) - graceNotesDuration;
				if (noteDuration <= 0)
				    noteDuration = getNoteLengthInTicks(note, staff);
				if (fermata) noteDuration *= 2;
				playNote(note, i, currentKey, elapsedTime, noteDuration, pitchClassHistogram);
				elapsedTime+=noteDuration;
			    }
			    else
				if ((voice.elementAt(i) instanceof abc.notation.MultiNote)) {
				    MultiNote multiNote = (MultiNote)voice.elementAt(i);
				    playMultiNote(multiNote, i, currentKey, elapsedTime, staff, pitchClassHistogram);
				    elapsedTime+=getNoteLengthInTicks(multiNote, staff);
				}
		    }
		    if (voice.elementAt(i) instanceof abc.notation.RepeatBarLine) {
			RepeatBarLine bar = (RepeatBarLine)voice.elementAt(i);
			if (repeatNumber<bar.getRepeatNumbers()[0] && lastRepeatOpen!=-1) {
			    repeatNumber++;
			    i=lastRepeatOpen;
			}
			else
			    if (repeatNumber>bar.getRepeatNumbers()[0])
				inWrongEnding = true;
			    else
				inWrongEnding = false;
		    }
		    else
			if (voice.elementAt(i) instanceof abc.notation.BarLine) {
			    switch ( ((BarLine)(voice.elementAt(i))).getType()) {
			    case BarLine.SIMPLE : break;
			    case BarLine.REPEAT_OPEN : lastRepeatOpen=i; repeatNumber=1; break;
			    case BarLine.REPEAT_CLOSE :
				if (repeatNumber<2 && lastRepeatOpen!=-1) {
				    repeatNumber++; i=lastRepeatOpen;
				}
				else {
				    repeatNumber=1; lastRepeatOpen=-1;
				}
				break;
			    }
			}
		    if (voice.elementAt(i) instanceof abc.notation.BarLine) {
			currentKey = new KeySignature(tuneKey.getAccidentals());
		    }
		    i++;
		}
	    }
	    return pitchClassHistogram;
	}
	catch (Exception e) {
	    e.printStackTrace();
	    return null;
	}
    }
    protected static void playNote(Note note, int indexInScore, KeySignature currentKey,
				   long timeReference,
				   long duration, long[] histogram) {
	if (!note.isRest() && !note.isEndingTie()) {
	    int noteNumber = getMidiNoteNumber (note, currentKey);
	    // System.out.println("Adding " + noteNumber + ", dur " + duration);
	    histogram[noteNumber % 12] += duration;
	    updateKey(currentKey, note);
	}
    }

    protected static void playMultiNote(MultiNote multiNote, int indexInScore, KeySignature currentKey,
					long reference, Music staff, long[] histogram)
    {
	Vector notesVector = multiNote.getNotesAsVector();
	for (int j=0; j<notesVector.size(); j++)
	    {
		Note note = (Note)(notesVector.elementAt(j));
		long noteDuration = getNoteLengthInTicks(multiNote, staff);
		int noteNumber = getMidiNoteNumber (note, currentKey);
		if (!note.isRest() && !note.isEndingTie()) {
		    // System.out.println("Adding " + noteNumber + ", dur " + noteDuration);
		    histogram[noteNumber%12] += noteDuration;
		}
	    }
	for (int j=0; j<notesVector.size(); j++)
	    updateKey(currentKey, (Note)notesVector.elementAt(j));
    }

    private static void updateKey(KeySignature key, Note note)
    {
	if (!note.getAccidental().isInTheKey()) {
	    key.setAccidental(note.getStrictHeight(), note.getAccidental());
	}
    }

    protected static long getNoteLengthInTicks(NoteAbstract note, Music staff) {
	if (note instanceof Note)
	    return getNoteLengthInTicks((Note)note, staff);
	else
	    return getNoteLengthInTicks((MultiNote)note, staff);
    }

    protected static long getNoteLengthInTicks(Note note, Music staff) {
	short noteLength = note.getDuration();
	if (note.isBeginningTie() && note.getTieDefinition().getEnd()!=null) {
	    try {
		noteLength +=
		    ((Note)staff.getElementByReference(note.getTieDefinition().getEnd()))
		    .getDuration();
	    } catch (Exception e) {}
	}
	float numberOfQuarterNotesInThisNote = (float)noteLength / Note.QUARTER;
	float lengthInTicks = (float)SEQUENCE_RESOLUTION * numberOfQuarterNotesInThisNote;
	return (long)lengthInTicks;
    }

    public static long getNoteLengthInTicks(MultiNote note, Music staff) {
	Note[] notes = note.toArray();
	notes = MultiNote.excludeTiesEndings(notes);
	if (notes!=null)
	    return getNoteLengthInTicks(note.getShortestNote(), staff);
	else
	    return 0;
    }

    public static int getMidiNoteNumber (Note note, KeySignature key)
    {
	byte heigth = note.getStrictHeight();
	Accidental accidental = new Accidental(note.getAccidental(key).getNearestOccidentalValue());
	byte midiNoteNumber = (byte)(heigth+(69-Note.A));
	midiNoteNumber = (byte)(midiNoteNumber + note.getOctaveTransposition()*12);
	midiNoteNumber += (byte) accidental.getValue();
	return (int)midiNoteNumber;
    }
}

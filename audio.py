from essentia.standard import MonoLoader, TuningFrequencyExtractor, \
    PitchMelodia, PitchContourSegmentation, EqualLoudness, \
    PredominantPitchMelodia
import madmom
import essentia
import librosa
from math import log
import numpy as np
import scipy.ndimage.filters as fi
import os.path
import json
import csv
import vamp
import sys
import pysptk
import matplotlib.pyplot as plt
# import crepe

class AudioProc():

    def __init__(self, audiopath, respath):
        self.filename = None
        self.index = None
        self.sig = None
        self.path = audiopath
        self.respath = respath

    def setFilename(self, f, index = None, offset = 0, duration = None):
        self.filename = f
        self.index = index

    def setOffset(self, offset, duration):
        self.offset = offset
        self.duration = duration
        self.sig = None

    def getSignal(self):
        # Using librosa to load audio - allows to set duration / offset
        # print 'reading audio', self.filename
        self.sig, _ = librosa.load(self.filename, sr=44100,
                                   offset = self.offset,
                                   duration = self.duration,
                                   mono = True)

    @classmethod
    def freq2midi(_, freq, tuning):
        if freq == 0:
            return 0
        return 12 * log(freq / tuning, 2) + 69

    def getODM(self, method, overwrite = False):
        basefile = self.respath + '/' + method + '/' + ("%03d" % self.index) + \
                      ".%d" % self.offset
        contourfile = basefile + ".contour.csv"
        notefile = basefile + ".notes.csv"
        # if os.path.isfile(contourfile) and \
        #    os.path.isfile(notefile) and not overwrite:
        #     print('\treading transcription from file')
            # (o, d, m) = ([], [], [])
            # t = None
            # with open(outfile, 'r') as cached:
            #     for line in cached:
            #         if t is None:
            #             t = float(line)
            #             continue
            #         v = line.split(', ')
            #         o.append(float(v[0]))
            #         d.append(float(v[1]))
            #         m.append(int(v[2]))
            # return outfile, o, d, m, t

        if self.sig is None:
            self.getSignal()

        # get tuning - return MODE of predictions
        tuner = TuningFrequencyExtractor()
        tf = tuner(self.sig)
        tc = {}
        for c in tf:
            if c not in tc:
                tc[c] = 0
            tc[c] += 1
            tuning = sorted( [(k, tc[k]) for k in tc],
                             key = lambda x: x[1],
                             reverse = True)[0][0]

        if method == 'pitchmelodia':
            elFilter = EqualLoudness()
            audio = elFilter(self.sig)

            melodia = PitchMelodia()
            contour, confidence = melodia(audio)

            segmenter = PitchContourSegmentation(tuningFrequency = int(tuning))
            o, d, m = segmenter(contour, audio)

        elif method == 'melodia':
            elFilter = EqualLoudness()
            audio = elFilter(self.sig)

            melodia = PredominantPitchMelodia()
            contour, confidence = melodia(audio)

            segmenter = PitchContourSegmentation(tuningFrequency = int(tuning))
            o, d, m = segmenter(contour, audio)

        elif method == 'silvet':
            notes = vamp.collect(self.sig, 44100, "silvet:silvet")['list']
            (o, d, m) = ([], [], [])
            for note in notes:
                o.append(note['timestamp'])
                d.append(note['duration'])
                m.append(AudioProc.freq2midi(note['values'][0], tuning))
            contour = None

        elif method == 'pyin':
            contour = vamp.collect(self.sig, 44100, "pyin:pyin", output="smoothedpitchtrack")['vector'][1]
            notes = vamp.collect(self.sig, 44100, "pyin:pyin", output='notes')['list']
            (o, d, m) = ([], [], [])
            for note in notes:
                o.append(note['timestamp'])
                d.append(note['duration'])
                m.append(AudioProc.freq2midi(note['values'][0], tuning))
            # contour = None

        elif method == 'swipe':
            contour = pysptk.sptk.swipe(self.sig.astype('float64'),
                                        44100, 128, max=2000)
            segmenter = PitchContourSegmentation(tuningFrequency = int(tuning))
            o, d, m = segmenter(essentia.array(contour), self.sig)
            m[np.where(np.isneginf(m))] = 0
            m[np.where(np.isnan(m))] = 0

        elif method == 'crepe':
            t_, contour, c_, a_ = crepe.predict(self.sig, 44100,
                                                viterbi=True,
                                                verbose=False)
            # contour = []
            # with open(contourfile, 'r') as infile:
            #     for line in infile:
            #         contour.append(float(line))
            segmenter = PitchContourSegmentation(tuningFrequency = int(tuning), hopSize = 441)
            o, d, m = segmenter(essentia.array(contour), self.sig)

        else:
            sys.exit(1)

        if contour is not None:
            np.savetxt(contourfile, contour, fmt = "%.3f")

        with open(notefile, 'w') as output:
            output.write("%f\n" % tuning)
            for t in range(len(o)):
                output.write('%.3f, %.3f, %.2f\n' % (o[t], d[t], m[t]))

        # return notefile, o, d, map(int, m), tuning
        return None, None, None, None, None

    @classmethod
    def processSpectrum(_, spectrum, freqs):
        def freq2midi(freq):
            if freq == 0:
                return 0
            return round(1200 * log(freq / 440., 2) + 6900)
        toCents = np.vectorize(freq2midi, otypes = [np.int])
        cents = toCents(freqs)
        maxCent = np.max(cents)
        counters = np.zeros(maxCent + 1)
        result = np.zeros(maxCent + 1)
        for c in range(len(cents)):
            if cents[c] <= 0:
                continue
            result[ cents[c] ] += spectrum[c]
            counters[ cents[c] ] += 1
        counters[ np.where(counters == 0) ] = 1
        result /= counters
        g = fi.gaussian_filter1d(result, 15, mode = 'wrap')
        ow = np.zeros(1200)
        for c in range(maxCent):
            ow[c % 1200] += g[c]
        ow120 = ow[ range(0, 1200, 10) ]
        s = sum(ow120)
        ow120 /= s
        return ow120.tolist()

    def getHPCP(self):
        s = madmom.audio.signal.Signal(self.filename, sample_rate = 44100, num_channels = 1,
                                       start = self.offset, stop = self.offset + 12)
        fs = madmom.audio.signal.FramedSignal(s, frame_size = 4096)
        stfs = madmom.audio.stft.ShortTimeFourierTransform(fs)
        spec = madmom.audio.spectrogram.Spectrogram(stfs)
        hpcp = madmom.audio.chroma.HarmonicPitchClassProfile(spec, num_classes = 120)
        print hpcp.shape
        
        hpcp = np.sum(hpcp, axis = 0)
        s = sum(hpcp)
        hpcp /= s
        plt.plot(hpcp)
        plt.show()
        sys.exit(0)
        return hpcp

    def getPCH(self, method, overwrite = False):

        outfile = self.respath + '/pch/' + ("%03d" % self.index) + \
                  (".%d.%s.json" % (self.offset, method))
        if os.path.isfile(outfile) and not overwrite:
            print('reading', outfile)
            with open(outfile, 'r') as datafile:
                return outfile, json.load(datafile)

        if self.sig is None:
            self.getSignal()

        if method == 'global':
            rspec = np.fft.rfft(self.sig)
            spectrum = np.abs(rspec)
            freqs = np.fft.rfftfreq(len(self.sig), d = 1./44100)
            cutOff = 0
            for f in range(len(freqs)):
                if freqs[f]>5000:
                    cutOff = f
                    break
            freqs = freqs[:f]
            spectrum = spectrum[:f]
        elif method == 'local':
            stft = np.abs(librosa.stft(self.sig, n_fft = 4096))
            spectrum = np.sum(stft,axis = 1)
            freqs = librosa.fft_frequencies(sr=44100, n_fft = 4096)
        elif method == '5k':
            stft = np.abs(librosa.stft(self.sig, n_fft = 4096))
            spectrum = np.sum(stft,axis = 1)
            freqs = librosa.fft_frequencies(sr=44100, n_fft = 4096)
            cutOff = 0
            for f in range(len(freqs)):
                if freqs[f]>5000:
                    cutOff = f
                    break
            freqs = freqs[:f]
            spectrum = spectrum[:f]
        elif method == 'deepChroma':
            dcp = madmom.audio.chroma.DeepChromaProcessor()
            chroma = dcp(self.sig)
            # print chroma.shape
            chroma = np.sum(chroma, axis = 0)
            # print chroma.shape
            pch1 = np.zeros((120))
            for i in range(12):
                pch1[i*10] = chroma[i]
            pch = fi.gaussian_filter1d(pch1, 15, mode = 'wrap')
            s = sum(pch)
            pch /= s
        elif method == 'HPCP':
            pch = self.getHPCP()
        else:
            print('getPCH(): method should be \'local\' or \'global\'')
            return None

        # pch = AudioProc.processSpectrum(spectrum, freqs)
        with open(outfile, 'w') as datafile:
            json.dump(pch, datafile)
        return outfile, pch

if __name__ == '__main__':
    import sys
    import glob

    path = 'soundfiles'
    transcriptpath = 'transcriptions'
    ap = AudioProc(path, transcriptpath)

    # algo = sys.argv[2]

    with open(sys.argv[1], 'r') as refFile:
        reader = csv.DictReader(refFile, delimiter = ',')
        for row in reader:
            print('processing tune', row['index'])
            ap.setFilename('%s/%s' % (path, row['file']),
                           index = int(row['index']))
            for t in ['t1', 't2', 't3', 't4']:
                print('\tfrom offset', row[t])
                ap.setOffset(int(row[t]), 12)
                # notes_f, _, _, _, _ = ap.getODM(algo, overwrite = True)
                pch_f, _ = ap.getPCH('HPCP', overwrite = True)

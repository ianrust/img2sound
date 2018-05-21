import numpy as np
import scipy.ndimage
import scipy.misc
import scipy.signal
import scipy.io.wavfile
import matplotlib.pyplot as plt
import pyaudio
import audio
import serial

duration = 0.03/80.0*60.0
fs = 44100

knit_ratio = 0.1
knit = int(fs*knit_ratio*duration)

ramp = np.arange(knit)/float(knit-1)
inverse_ramp = 1.0-ramp

loops = 1

factor = np.power(1.05946, 1.0/8.0)
start_freq = 130.82/4.0
 
f = np.array([
130.82,
138.59,
146.83,
155.56,
164.81,
174.61,
185,
196,
207.65,
220,
233.08,
246.94,
])/4.0

def makeColumnWavRealtime(fn):
	image = scipy.ndimage.imread(fn, flatten=True)

	def spatial2Audio(freq, amplitude, size=(fs*duration)):
		octave = np.power(2.0, np.floor(freq/10.0))
		audio_freq = f[int(freq%10)]*octave
		angle = 2*np.pi*np.arange(size)*audio_freq/fs + last_phase[freq]
		sample = amplitude*((np.sin(angle))).astype(np.float32)
		last_phase[freq] = np.mod(angle[-1], 2*np.pi)

		return sample

	
	def getBand(size):
		if getBand.full_sound is None:
			return np.array([0]*size).astype(np.float32)
		else:
			out = getBand.full_sound[:size]
			getBand.full_sound = getBand.full_sound[size+1:]
			return out
	getBand.full_sound = None

	gs = lambda size: getBand(size)
	loop = audio.WavePlayerLoop(gs)
	loop.play()

	ser = serial.Serial('/dev/tty.usbmodem14511')
	while True:
		ser.reset_input_buffer()
		line = ser.readline().strip()
		line = ser.readline().strip()
		print line
		if line.isdigit():
			print int(line)
			u = int(line)%image.shape[1]
			print str(u) + "/" + str(image.shape[1])
			col_sound = None
			for v in range(image.shape[0]):
				amplitude = image[v, u]
				frequency = v
				if col_sound is None:
					col_sound = spatial2Audio(frequency, amplitude, 1024*3)
				else:
					col_sound += spatial2Audio(frequency, amplitude, 1024*3)
			if getBand.full_sound is None:
				getBand.full_sound = col_sound
			else:
				# knit_portion = np.multiply(full_sound[-knit:], inverse_ramp) + np.multiply(col_sound[:knit], ramp)
				# full_sound = np.append(full_sound[:-knit], knit_portion)
				# full_sound = np.append(full_sound, col_sound[knit:])
				getBand.full_sound = np.append(getBand.full_sound, col_sound)



def makeColumnWav(fn):
	image = scipy.ndimage.imread(fn, flatten=True)

	# last_phase = np.zeros(len(f)*100)
	last_phase = np.random.random(len(f)*100)*2*np.pi
	# print last_phase

	def spatial2Audio(freq, amplitude):
		# piano_freq = freq
		# octave = np.power(2.0, np.floor(piano_freq/float(len(f))))
		# audio_freq = f[piano_freq%len(f)]*octave
		audio_freq = start_freq*np.power(factor, freq)
		# print audio_freq
		angle = 2*np.pi*np.arange(fs*duration)*audio_freq/fs + last_phase[freq]
		sample = amplitude*((np.sin(angle))).astype(np.float32)/(audio_freq)
		# sample = amplitude*((scipy.signal.square(angle))).astype(np.float32)/(audio_freq)
		# sample = amplitude*((scipy.signal.sawtooth(angle))).astype(np.float32)/(audio_freq)
		last_phase[freq] = np.mod(angle[-knit], 2*np.pi)

		return sample*np.power(freq+1.0,0.9)

	full_sound = None
	for l in range(0,loops):
		for u in range(image.shape[1]):
			print str(u) + "/" + str(image.shape[1])
			col_sound = None
			for v in range(image.shape[0]):
				amplitude = image[v, u]
				frequency = v
				if col_sound is None:
					col_sound = spatial2Audio(frequency, amplitude)
				else:
					col_sound += spatial2Audio(frequency, amplitude)
			if full_sound is None:
				full_sound = col_sound
			else:
				knit_portion = np.add(np.multiply(full_sound[-knit:], inverse_ramp), np.multiply(col_sound[:knit], ramp))
				full_sound = np.append(full_sound[:-(knit+1)], knit_portion)
				full_sound = np.append(full_sound, col_sound[(knit+1):])
				# full_sound = np.append(full_sound, col_sound)
				# plt.figure(0)
				# plt.clf()
				# # plt.plot(np.multiply(full_sound[-knit:], inverse_ramp))
				# # plt.plot(np.multiply(col_sound[:knit], ramp))
				# plt.plot(full_sound[-int(len(col_sound)*1.5):])
				# plt.plot(knit_portion)
				# plt.plot(col_sound)
				# plt.draw()
				# plt.pause(100)

		# plt.figure(0)
		# plt.clf()
		# plt.plot(col_sound[:100])
		# plt.plot(full_sound[(-fs*duration):(-fs*duration+100)])
		# plt.draw()
		# plt.pause(1)
		# plt.figure(1)
		# plt.clf()
		# plt.plot(full_sound[(-fs*duration-100):(-fs*duration+100)])
		# plt.draw()
		# plt.figure(2)
		# plt.clf()
		# plt.plot(full_sound[(-fs*duration-100):(-fs*duration)])
		# plt.draw()
		# plt.pause(2)
		# plt.clf()
		# plt.plot(full_sound)
		# plt.draw()
		# plt.pause(1)

	fade = int(1.0*fs)
	ramp_mult = np.arange(fade)
	ramp_mult = ramp_mult/float(fade-1)
	full_sound[0:fade] = np.multiply(full_sound[0:fade], ramp_mult)
	fade = int(5*fs)
	ramp_mult = np.arange(fade)
	ramp_mult = ramp_mult/float(fade-1)
	full_sound[-fade:] = np.multiply(full_sound[-fade:], 1.0-ramp_mult)

	scipy.io.wavfile.write(fn.split(".")[0]+".wav",fs,full_sound/np.max(full_sound))




def makeFFTWav(fn):
	image = scipy.ndimage.imread(fn, flatten=True)

	# plt.imshow(np.abs(image), cmap='gray')
	# plt.show()

	fft = np.fft.fft2(image, s=[20, 20])
	# fft = np.fft.fftshift(fft)

	u_range = 10
	v_range = 10

	content = []
	for u in range(1,u_range):
		for v in range(1,v_range):
			frequency = np.sqrt(u**2 + v**2)
			direction = np.arctan2(v,u)
			amplitude = np.abs(fft[u,v])
			phase = np.arctan2(np.imag(fft[u,v]), np.real(fft[u,v]))
			content.append((frequency, amplitude, phase, direction))

	all_frequencies = [c[0] for c in content]
	all_amplitudes = [c[1] for c in content]
	freq_scale = (max_freq-min_freq)/(np.max(all_frequencies) - np.min(all_frequencies))
	amplitude_scale = (4.0/float(len(content)))/(np.max(all_amplitudes) - np.min(all_amplitudes))

	# TODO - do something with direction
	def spatial2Audio(freq, amplitude, phase, direction):
		audio_freq = freq*freq_scale+min_freq
		audio_amplitude = amplitude*amplitude_scale
		sample = audio_amplitude*(np.sin(2*np.pi*np.arange(fs*duration)*audio_freq/fs)+phase).astype(np.float32)
		return sample

	audio = None
	for c in content:
		component_audio = spatial2Audio(*c)
		if audio is None:
			audio = component_audio
		else:
			audio += component_audio

	audio = audio/np.max(audio)
	# print np.max(audio)

	p = pyaudio.PyAudio()
	stream = p.open(format=pyaudio.paFloat32,
	                channels=1,
	                rate=fs,
	                output=True)

	stream.write(audio)

	stream.stop_stream()
	stream.close()

	p.terminate()

	scipy.io.wavfile.write(fn.split(".")[0]+".wav",fs,audio)



fn = [
		"flower.png",
		"kate.jpg",
		"mushroom.jpg",
		# "s1.jpeg",
		# "s2.jpeg",
		# "s3.jpeg"
	 ]

# map(makeFFTWav, fn)
map(makeColumnWav, fn)
# map(makeColumnWavRealtime, fn)
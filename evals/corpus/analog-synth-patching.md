# Analog Synth Patching

The modular rack in Studio B is the house sound. This guide covers the basic subtractive voice.

## Oscillators

The voltage-controlled oscillator (VCO) generates the raw waveform: saw, square, triangle or sine. Pitch follows the one volt per octave standard, so a keyboard sending 1 V more raises the pitch by exactly one octave. Let oscillators warm up for twenty minutes before tuning, because analog circuits drift as they heat.

## Filters

The voltage-controlled filter (VCF) removes harmonics. A low-pass filter darkens the tone as the cutoff frequency drops. Resonance boosts frequencies around the cutoff; at high resonance many filters self-oscillate and produce a sine wave of their own. The classic synthwave bass is a saw wave through a 24 dB per octave low-pass with moderate resonance.

## Envelopes

An envelope generator shapes a sound over time with four stages: attack, decay, sustain and release (ADSR). Attack is how long the sound takes to reach full level, decay is the fall to the sustain level, sustain is the level held while the key is pressed, and release is the fade after the key is let go. For plucky arpeggios use a fast attack, short decay and zero sustain.

## LFO modulation

A low-frequency oscillator (LFO) runs below the audible range and is used to modulate other parameters. Routing a slow triangle LFO to filter cutoff gives the familiar sweeping pad; routing it to pitch produces vibrato.

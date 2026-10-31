# MIDI Clock and Sync

Every sequencer, drum machine and arpeggiator in the studio follows one master clock.

## MIDI clock basics

MIDI clock sends 24 pulses per quarter note (24 PPQN). The receiving device derives tempo from the spacing of these pulses. Start, Stop and Continue messages control transport. Song Position Pointer lets a device jump to a specific bar when Continue is used.

## Choosing a master

Pick exactly one device as clock master and set all others to external sync. In our setup the DAW is master during tracking and the hardware sequencer is master during live jams. Two masters on the same chain cause tempo fights and double-triggered notes.

## Jitter and latency

USB MIDI interfaces can add clock jitter, which makes arpeggios feel loose. If timing sounds uneven, use a dedicated hardware MIDI interface or a clock distribution box, and avoid daisy-chaining more than three devices through MIDI thru ports.

## DIN sync and analog clock

Vintage drum machines use DIN sync, which runs at 24 PPQN on a five-pin DIN cable but is electrically different from MIDI. Some modular gear expects analog clock pulses at 1, 2 or 4 PPQN; a clock divider module converts between rates.

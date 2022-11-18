mkdir -p wavs_new
for i in wav_mono_44khz/*.wav; do
    o=wavs_new/${i#wavs/}
    sox "$i" -r 44100 "${o%.wav}.wav"
done
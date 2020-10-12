# vim:ts=4 sw=4 nowrap:
# Tempo que funcionou em 25fps: 4.45
ffmpeg -y -i ../trabalho1/camera1.webm.bkp -vf scale=1280:720 -ss 4.45 \
	-r 10 -vsync 2 -crf 28 -q 1 -c:v h264_videotoolbox -q:v 1 -b:v 10000k timestamp-cam-1.mp4
ffmpeg -y -i ../trabalho1/camera2.webm                       					   \
	-r 10 -vsync 2 -crf 28 -q 1 -c:v h264_videotoolbox -q:v 1 -b:v 10000k timestamp-cam-2.mp4

ffmpeg \
	-i timestamp-cam-1.mp4\
	-i timestamp-cam-2.mp4\
	-filter_complex \
		 "[0]drawtext=text=frame %{n}tempo %{pts}:fontsize=72:x=(w-tw)/2: y=h-(2*lh):fontcolor=white:box=1:boxcolor=0x00000099[0t];\
		  [1]drawtext=text=frame %{n}tempo %{pts}:fontsize=72:x=(w-tw)/2: y=h-(2*lh):fontcolor=white:box=1:boxcolor=0x00000099[1t];\
		  [1:a][0:a]join=inputs=2:channel_layout=stereo, asplit[stereo][stereo1];\
		  [1t][0t]hstack=shortest=1[v]" \
	-pix_fmt yuv420p -crf 1  -preset veryslow \
	-map '[stereo]' -map '[v]' \
	-c:v h264_videotoolbox -q:v 1 -b:v 1000k \
	-y concat.mp4\
	-map '[stereo1]' stereo.wav


#		 [0t]drawtext=timecode='00\:00\:00\:00':r=25:x=(w-tw)/2:y=h-(2*lh):fontcolor=white:box=1:boxcolor=0x00000000@1[0tt];\
#		[1t]drawtext=timecode='00\:00\:00\:00':r=25:x=(w-tw)/2:y=h-(2*lh):fontcolor=white:box=1:boxcolor=0x00000000@1[1tt];\

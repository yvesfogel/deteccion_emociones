import ffmpeg

vid = ffmpeg.probe("video.mp4")

def duracion_base(vid):
	return (float(vid['streams'][0]["duration"]))

def duracion_frames_total(vid):
	frames = duracion_base(vid)
	framerate = float((vid['streams'][0]["avg_frame_rate"]).split("/")[0])/1000
	frames *= framerate
	frames = int(frames)
	return frames

def duracion_segs_total(vid):
	#devuelve int con cantidad de segundos redondeado para abajo
	segundos = int(duracion_base(vid))
	return segundos

def duracion_mins_total(vid):
	base = duracion_base(vid)
	mins = int(base/60)
	return mins

def duracion_horas_total(vid):
	base = duracion_base(vid)
	horas = int((base/60)/60)
	return horas


def duracion_total(vid, horas_bool=False, mins_bool=False, segs_bool=True, frames_bool=False):
	#devuelve lista con cantidad de horas, minutos, segundos y frames

	horas_t = duracion_horas_total(vid)
	mins_t = duracion_mins_total(vid)
	segundos_t = duracion_segs_total(vid)
	frames_t = duracion_frames_total(vid)

	horas = 0
	mins = 0
	segundos = 0
	frames = 0

	lista = []

	if(horas_bool):
		horas = horas_t
		lista.append(int(horas))
	if (mins_bool):
		mins = mins_t - horas*60
		lista.append(int(mins))
	if(segs_bool):
		segundos = segundos_t - (mins*60 + horas*3600)
		lista.append(int(segundos))
	if(frames_bool):
		frames = frames_t - ((segundos + mins*60 + horas*3600)*float((vid['streams'][0]["avg_frame_rate"]).split("/")[0])/1000)
		lista.append(frames)

	return lista



print(duracion_total(vid))

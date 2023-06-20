"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from tempfile import NamedTemporaryFile
import argparse
import torch
import gradio as gr
import os
import time
import warnings
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import apply_fade, apply_tafade
from audiocraft.utils.extend import generate_music_segments, add_settings_to_image, INTERRUPTING
import numpy as np
import random

MODEL = None
MODELS = None
IS_SHARED_SPACE = "musicgen/MusicGen" in os.environ.get('SPACE_ID', '')
INTERRUPTED = False
UNLOAD_MODEL = False
MOVE_TO_CPU = False

def interrupt_callback():
    return INTERRUPTED

def interrupt():
    global INTERRUPTING
    INTERRUPTING = True


def make_waveform(*args, **kwargs):
    # Further remove some warnings.
    be = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = gr.make_waveform(*args, **kwargs)
        print("Make a video took", time.time() - be)
        return out

def load_model(version):
    global MODEL, MODELS, UNLOAD_MODEL
    print("Loading model", version)
    if MODELS is None:
        return MusicGen.get_pretrained(version)
    else:
        t1 = time.monotonic()
        if MODEL is not None:
            MODEL.to('cpu') # move to cache
            print("Previous model moved to CPU in %.2fs" % (time.monotonic() - t1))
            t1 = time.monotonic()
        if MODELS.get(version) is None:
            print("Loading model %s from disk" % version)
            result = MusicGen.get_pretrained(version)
            MODELS[version] = result
            print("Model loaded in %.2fs" % (time.monotonic() - t1))
            return result
        result = MODELS[version].to('cuda')
        print("Cached model loaded in %.2fs" % (time.monotonic() - t1))
        return result


def predict(model, text, melody, duration, dimension, topk, topp, temperature, cfg_coef, background, title, include_settings, settings_font, settings_font_color, seed, overlap=1, generate_audio_download=False):
    global MODEL, INTERRUPTED, INTERRUPTING
    output_segments = None

    INTERRUPTED = False
    INTERRUPTING = False
    if temperature < 0:
        raise gr.Error("Temperature must be >= 0.")
    if topk < 0:
        raise gr.Error("Topk must be non-negative.")
    if topp < 0:
        raise gr.Error("Topp must be non-negative.")

    if MODEL is None or MODEL.name != model:
        MODEL = load_model(model)
    else:
        if MOVE_TO_CPU:
            MODEL.to('cuda')
    
    # prevent hacking
    duration = min(duration, 720)
    overlap =  min(overlap, 15)
    # 

    output = None
    segment_duration = duration
    initial_duration = duration
    output_segments = []
    audio_file = None
    while duration > 0:
        if not output_segments: # first pass of long or short song
            if segment_duration > MODEL.lm.cfg.dataset.segment_duration:
                segment_duration = MODEL.lm.cfg.dataset.segment_duration
            else:
                segment_duration = duration
        else: # next pass of long song
            if duration + overlap < MODEL.lm.cfg.dataset.segment_duration:
                segment_duration = duration + overlap
            else:
                segment_duration = MODEL.lm.cfg.dataset.segment_duration
        # implement seed
        if seed < 0:
            seed = random.randint(0, 0xffff_ffff_ffff)
        torch.manual_seed(seed)


        print(f'Segment duration: {segment_duration}, duration: {duration}, overlap: {overlap}')
        MODEL.set_generation_params(
            use_sampling=True,
            top_k=topk,
            top_p=topp,
            temperature=temperature,
            cfg_coef=cfg_coef,
            duration=segment_duration,
            two_step_cfg=False,
            rep_penalty=0.5
        )

        if melody:
            # todo return excess duration, load next model and continue in loop structure building up output_segments
            if duration > MODEL.lm.cfg.dataset.segment_duration:
                output_segments, duration = generate_music_segments(text, melody, seed, MODEL, duration, overlap, MODEL.lm.cfg.dataset.segment_duration)
            else:
                # pure original code
                sr, melody = melody[0], torch.from_numpy(melody[1]).to(MODEL.device).float().t().unsqueeze(0)
                print(melody.shape)
                if melody.dim() == 2:
                    melody = melody[None]
                melody = melody[..., :int(sr * MODEL.lm.cfg.dataset.segment_duration)]
                output = MODEL.generate_with_chroma(
                    descriptions=[text],
                    melody_wavs=melody,
                    melody_sample_rate=sr,
                    progress=True
                )
            # All output_segments are populated, so we can break the loop or set duration to 0
            break
        else:
            #output = MODEL.generate(descriptions=[text], progress=False)
            if not output_segments:
                next_segment = MODEL.generate(descriptions=[text], progress=True)
                duration -= segment_duration
            else:
                last_chunk = output_segments[-1][:, :, -overlap*MODEL.sample_rate:]
                next_segment = MODEL.generate_continuation(last_chunk, MODEL.sample_rate, descriptions=[text], progress=True)
                duration -= segment_duration - overlap
            output_segments.append(next_segment)

        if INTERRUPTING:
            INTERRUPTED = True
            INTERRUPTING = False
            print("Function execution interrupted!")
            raise gr.Error("Interrupted.")

    if output_segments:
        try:
            # Combine the output segments into one long audio file or stack tracks
            #output_segments = [segment.detach().cpu().float()[0] for segment in output_segments]
            #output = torch.cat(output_segments, dim=dimension)
            
            output = output_segments[0]
            for i in range(1, len(output_segments)):
                overlap_samples = overlap * MODEL.sample_rate
                #stack tracks and fade out/in
                overlapping_output_fadeout = output[:, :, -overlap_samples:]
                overlapping_output_fadeout = apply_fade(overlapping_output_fadeout,sample_rate=MODEL.sample_rate,duration=overlap,out=True,start=True, curve_end=0.0, current_device=MODEL.device)
                #overlapping_output_fadeout = apply_tafade(overlapping_output_fadeout,sample_rate=MODEL.sample_rate,duration=overlap,out=True,start=True,shape="exponential")

                overlapping_output_fadein = output_segments[i][:, :, :overlap_samples]
                overlapping_output_fadein = apply_fade(overlapping_output_fadein,sample_rate=MODEL.sample_rate,duration=overlap,out=False,start=False, curve_start=0.0, current_device=MODEL.device)
                #overlapping_output_fadein = apply_tafade(overlapping_output_fadein,sample_rate=MODEL.sample_rate,duration=overlap,out=False,start=False, shape="linear")

                overlapping_output = torch.cat([overlapping_output_fadeout[:, :, :-(overlap_samples // 2)], overlapping_output_fadein],dim=2)
                print(f" overlap size Fade:{overlapping_output.size()}\n output: {output.size()}\n segment: {output_segments[i].size()}")
                ##overlapping_output = torch.cat([output[:, :, -overlap_samples:], output_segments[i][:, :, :overlap_samples]], dim=1) #stack tracks
                ##print(f" overlap size stack:{overlapping_output.size()}\n output: {output.size()}\n segment: {output_segments[i].size()}")
                #overlapping_output = torch.cat([output[:, :, -overlap_samples:], output_segments[i][:, :, :overlap_samples]], dim=2) #stack tracks
                #print(f" overlap size cat:{overlapping_output.size()}\n output: {output.size()}\n segment: {output_segments[i].size()}")               
                output = torch.cat([output[:, :, :-overlap_samples], overlapping_output, output_segments[i][:, :, overlap_samples:]], dim=dimension)
            output = output.detach().cpu().float()[0]
        except Exception as e:
            print(f"Error combining segments: {e}. Using the first segment only.")
            output = output_segments[0].detach().cpu().float()[0]
    else:
        output = output.detach().cpu().float()[0]
    
    with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
        if include_settings:
            video_description = f"{text}\n Duration: {str(initial_duration)} Dimension: {dimension}\n Top-k:{topk} Top-p:{topp}\n Randomness:{temperature}\n cfg:{cfg_coef} overlap: {overlap}\n Seed: {seed}\n Model: {model}\n Melody File:#todo"
            background = add_settings_to_image(title, video_description, background_path=background, font=settings_font, font_color=settings_font_color)
        audio_file = audio_write(
            file.name, output, MODEL.sample_rate, strategy="loudness",
            loudness_headroom_db=18, loudness_compressor=True, add_suffix=False, channels=2)
        waveform_video = make_waveform(file.name,bg_image=background, bar_count=45)
    if MOVE_TO_CPU:
        MODEL.to('cpu')
    if UNLOAD_MODEL:
        MODEL = None
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return waveform_video, seed, audio_file if generate_audio_download else None


def ui(**kwargs):
    css="""
    #col-container {max-width: 910px; margin-left: auto; margin-right: auto;}
    a {text-decoration-line: underline; font-weight: 600;}
    """
    with gr.Blocks(title="Jam Generator", theme="darkdefault", css=css) as interface:
        gr.Markdown(
            """
            # Jam Generator Limited Features Demo
            """
        )
        if IS_SHARED_SPACE and not torch.cuda.is_available():
            gr.Markdown("""
                ⚠ This Space doesn't work in this shared UI ⚠
                """)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Input Text", interactive=True, value="4/4 100bpm 320kbps 48khz, Industrial/Electronic Soundtrack, Dark, Intense, Sci-Fi")
                    melody = gr.Audio(source="upload", type="numpy", label="Melody Condition (optional)", interactive=True)
                with gr.Row():
                    submit = gr.Button("Submit")
                    # Adapted from https://github.com/rkfg/audiocraft/blob/long/app.py, MIT license.
                    _ = gr.Button("Interrupt").click(fn=interrupt, queue=False)
                with gr.Row():
                    background= gr.Image(value="./assets/sd_cyberpunk_hyper-futuristic_neon_rave.jpg", source="upload", label="Choose an Image for the Video", shape=(768,512), type="filepath", interactive=True)
                    include_settings = gr.Checkbox(label="Add Configuration to image", value=True, interactive=True)
                    generate_audio_download = gr.Checkbox(label="Generate Downloadable Audio (mp3)", value=False, interactive=True)
                with gr.Row():
                    title = gr.Textbox(label="Title", value="MusicGen", interactive=True)
                    settings_font = gr.Text(label="Settings Font", value="./assets/arial.ttf", interactive=True)
                    settings_font_color = gr.ColorPicker(label="Settings Font Color", value="#c87f05", interactive=True)
                with gr.Row():
                    model = gr.Radio(["melody", "medium", "small", "large"], label="Model", value="melody", interactive=True)
                with gr.Row():
                    duration = gr.Slider(minimum=1, maximum=720, value=10, label="Duration", interactive=True)
                    overlap = gr.Slider(minimum=2, maximum=14, value=2, step=2, label="Overlap", interactive=True)
                    dimension = gr.Slider(minimum=-2, maximum=2, value=2, step=1, label="Dimension", info="determines which direction to add new segements of audio. (1 = stack tracks, 2 = lengthen, -2..0 = ?)", interactive=True)
                with gr.Row():
                    topk = gr.Number(label="Top-k", value=250, precision=0, interactive=True)
                    topp = gr.Number(label="Top-p", value=0, precision=0, interactive=True)
                    temperature = gr.Number(label="Randomness Temperature", value=0.75, precision=None, interactive=True)
                    cfg_coef = gr.Number(label="Classifier Free Guidance", value=5.5, precision=None, interactive=True)
                with gr.Row():
                    seed = gr.Number(label="Seed", value=-1, precision=0, interactive=True)
                    gr.Button('\U0001f3b2\ufe0f\u000d Randomize Seed ').style(full_width=False).click(fn=lambda: -1, outputs=[seed], queue=False)
                    reuse_seed = gr.Button('\u267b\ufe0f\u000d Reuse Seed ').style(full_width=False)
            with gr.Column() as c:
                output = gr.Video(label="Generated Music")
                seed_used = gr.Number(label='Seed used', value=-1, interactive=False)
                if generate_audio_download:
                    audio_file = gr.File(label="Downloadable Audio (mp3)", type="auto", source="download", interactive=False)
                else:
                    audio_file = None

        reuse_seed.click(fn=lambda x: x, inputs=[seed_used], outputs=[seed], queue=False)
        submit.click(predict, inputs=[model, text, melody, duration, dimension, topk, topp, temperature, cfg_coef, background, title, include_settings, settings_font, settings_font_color, seed, overlap, generate_audio_download], outputs=[output, seed_used, audio_file ])
        gr.Examples(
            fn=predict,
            examples=[
                [
                    "4/4 120bpm 320kbps 48khz, An 80s driving pop song with heavy drums and synth pads in the background",
                    "./assets/bach.mp3",
                    "melody"
                ],
                [
                    "4/4 120bpm 320kbps 48khz, A cheerful country song with acoustic guitars",
                    "./assets/bolero_ravel.mp3",
                    "melody"
                ],
                [
                    "4/4 120bpm 320kbps 48khz, 90s rock song with electric guitar and heavy drums",
                    None,
                    "medium"
                ],
                [
                    "4/4 120bpm 320kbps 48khz, a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions",
                    "./assets/bach.mp3",
                    "melody"
                ],
                [
                    "4/4 320kbps 48khz, lofi slow bpm electro chill with organic samples",
                    None,
                    "medium",
                ],
            ],
            inputs=[text, melody, model],
            outputs=[output]
        )
        gr.Markdown(
            """

            ## Settings Explanation

            * Top-k: Top-k is a parameter used in text generation models, including music generation models. It determines the number of most likely next tokens to consider at each step of the generation process. The model ranks all possible tokens based on their predicted probabilities, and then selects the top-k tokens from the ranked list. The model then samples from this reduced set of tokens to determine the next token in the generated sequence. A smaller value of k results in a more focused and deterministic output, while a larger value of k allows for more diversity in the generated music.
            * Top-p (or nucleus sampling): Top-p, also known as nucleus sampling or probabilistic sampling, is another method used for token selection during text generation. Instead of specifying a fixed number like top-k, top-p considers the cumulative probability distribution of the ranked tokens. It selects the smallest possible set of tokens whose cumulative probability exceeds a certain threshold (usually denoted as p). The model then samples from this set to choose the next token. This approach ensures that the generated output maintains a balance between diversity and coherence, as it allows for a varying number of tokens to be considered based on their probabilities.
            * Temperature: Temperature is a parameter that controls the randomness of the generated output. It is applied during the sampling process, where a higher temperature value results in more random and diverse outputs, while a lower temperature value leads to more deterministic and focused outputs. In the context of music generation, a higher temperature can introduce more variability and creativity into the generated music, but it may also lead to less coherent or structured compositions. On the other hand, a lower temperature can produce more repetitive and predictable music.
            * Classifier-Free Guidance: Classifier-Free Guidance refers to a technique used in some music generation models where a separate classifier network is trained to provide guidance or control over the generated music. This classifier is trained on labeled data to recognize specific musical characteristics or styles. During the generation process, the output of the generator model is evaluated by the classifier, and the generator is encouraged to produce music that aligns with the desired characteristics or style. This approach allows for more fine-grained control over the generated music, enabling users to specify certain attributes they want the model to capture.
            
            Together, these parameters provide different ways of influencing how the model generates the audio track and allow a user to strike a balance between creativity, diversity, coherence, and control. 
            The specific values for these parameters can be tuned based on the desired outcome and user preferences.

            ## Prompting Tips
            
            * Include beats per minute, e.g. _120bpm_. Typical dance songs are 120. Hip-hop is often in the range of 90-110. A super slow song might be somewhere in the 70-90 range. 140+ is most often used in EDM
            * Include the bitrate and sample rate, e.g. _320kbps 48khz_. These values describe the sound quality of the generated audio track. Specifically, 320kbps and 48khz represent higher quality quality tracks with reduced hissing and, generally speaking, an expanded sound range. These numbers are on the higher end for an MP3 file, but they are not near the values of a raw audio recording. 
                * *NOTE:* Don't apply these specific values to tracks that are supposed to be LOFI, as lo-fi literally stands for low-fidelity, and 320kbps 48khz represent high-fidelity audio. 
            * Include time signatures, e.g. _4/4_, _3/4_, _5/4_, _2/4_, _etc_. Without getting too lost in the details, these numbers represent how many beats per measure (which is essentially the sections separated by vertical bars in sheet music) and the types of notes that each measure contains. 
                * The important thing to know here is that the most common time signatures are _4/4_ and _3/4_, with the vast majority being _4/4_ in modern music. Slow jams & waltzes are typically _3/4_.
            * Include a genre, e.g. _rock_, _pop_, _hip-hop_, _classical_, _jazz_, _etc_. This is a very broad category, but it helps the model understand what kind of instruments to use and what kind of sounds to generate.
            * Include a mood, e.g. _happy_, _sad_, _dark_, _intense_, _etc_. This is another broad category, but it helps the model understand what kind of instruments to use and what kind of sounds to generate.
            * Examples:
                * 4/4 100bpm 320kbps 48khz motown groove
                * 3/4 105bpm 320kbps 48khz piano only baroque
                * 110bpm 64kbps 16khz lofi hiphop summer smooth

            ### More details

            * The model will generate a short music extract based on the description you provided.
            * The model can generate up to 30 seconds of audio _in one pass_. 
                * The generation is extended by feeding back the last part of the previous chunk of audio.
                * This can take a long time, and the model might lose consistency. 
                * The model might also decide at arbitrary positions that the song ends.
            * If you want just the audio, make sure you select the checkbox for "Generate Downloadable Audio (mp3)". The audio track will be extracted from the video using an ffmpeg subprocess and saved to a downloadable file.
            **PLEASE NOTE:** Choosing long durations will take a long time to generate (2min might take ~10min). You 
            can choose the duration of the overlapping audio chunks with the `overlap` slider.

            We present 4 model variations:
            1. Melody -- a music generation model capable of generating music condition on text and melody inputs. **Note**, you can also use text only.
            2. Small -- a 300M transformer decoder conditioned on text only.
            3. Medium -- a 1.5B transformer decoder conditioned on text only.
            4. Large -- a 3.3B transformer decoder conditioned on text only (might OOM for the longest sequences.)

            When using `melody`, you can optionally provide a reference audio from
            which a broad melody will be extracted. The model will then try to follow both the description and melody provided.
            """
        )

        # Show the interface
        launch_kwargs = {}
        username = kwargs.get('username')
        password = kwargs.get('password')
        server_port = kwargs.get('server_port', 0)
        inbrowser = kwargs.get('inbrowser', False)
        share = kwargs.get('share', False)
        server_name = kwargs.get('listen')

        launch_kwargs['server_name'] = server_name

        if username and password:
            launch_kwargs['auth'] = (username, password)
        if server_port > 0:
            launch_kwargs['server_port'] = server_port
        if inbrowser:
            launch_kwargs['inbrowser'] = inbrowser
        if share:
            launch_kwargs['share'] = share

        interface.queue().launch(**launch_kwargs, max_threads=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=7859,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )

    args = parser.parse_args()

    ui(
        username=args.username,
        password=args.password,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        share=args.share,
        listen=args.listen
    )

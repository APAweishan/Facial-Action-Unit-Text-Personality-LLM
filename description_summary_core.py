import os
from typing import Callable, Iterable, List, Optional


EXAMPLE_ALL = """
Input:
description1:
From frames 68 to 74, the inner corners of the eyebrows start slightly lifted but steadily lower until they return to a neutral position, while the outer parts of the eyebrows show a pronounced rise at the beginning which gradually softens throughout. The eyebrows are pulled slightly together, forming faint vertical wrinkles that remain consistent with subtle variation. The upper lip does not show any significant movement, remaining relaxed throughout the sequence, and the cheeks are lifted moderately, leading to noticeable deepening of the lines beneath the eyes. The lower eyelids are markedly tightened, causing substantial narrowing of the eye aperture, though this dynamic stays fairly stable across the frames. The nose remains unmoved with no wrinkling or lifting of the nasal wings observed. The upper lip and lip corners show no activity or significant changes, remaining neutral and relaxed. There is no inward tightening or downward pull of the lip corners, while minor chin wrinkling gradually increases as the frames progress. The corners of the lips are pulled outward slightly at first with faint tension that lessens later, with no noticeable pressing or bulging. The lips part slightly, exposing the teeth intermittently, while the jaw shows a moderate lowering that increases briefly and stabilizes toward the end. The eyelids remain open without any abrupt closing.
description2:
From frames 97 to 103, the inner corners of the eyebrows are initially slightly lifted, with a moderate increase in elevation toward the end. The outer parts of the eyebrows show a steady and pronounced rise throughout, which becomes more prominent in the final frames. The eyebrows are drawn together lightly, and there is subtle vertical wrinkling at the glabella area that grows progressively deeper. The upper lip remains stable with no upward movement, while the cheeks are consistently raised, causing a significant deepening of the infraorbital furrow and an increase in wrinkles beneath the eyes. The lower eyelids are markedly lifted throughout, narrowing the eye aperture and creating visible tension. The nasal wings show only faint upward wrinkling at the very end, while the upper lip begins with minor lifting and quickly settles into an unchanged position. The lip corners remain stable and show no noticeable upward or inward movement, with no downward pull appearing either. The chin shows steady wrinkling accompanied by subtle upward pressing of the lower lip, which diminishes gradually. The corners of the lips display a brief outward horizontal pull early on, but this effect weakens and disappears later in the sequence. The lips remain relaxed and unpressed throughout, without visible bulging or tension. The lips do not part significantly, with minimal jaw lowering only becoming evident in the final frames. The eyelids remain open, with no abrupt closures or blinking observed.
Output:
From frames 68 to 103, the eyebrows exhibit a dynamic interplay: the inner corners begin slightly lifted, dip briefly, and then rise moderately toward the end, while the outer parts show a pronounced and steadily increasing elevation. The brows are gently drawn together, producing subtle vertical wrinkles at the glabella that deepen progressively. The cheeks maintain a consistent lift, enhancing the infraorbital furrows and deepening lines beneath the eyes, while the lower eyelids remain tight, narrowing the eye aperture with steady tension. The nose stays largely neutral, with only faint upward wrinkling at the nasal wings toward the final frames. The upper lip remains mostly relaxed, with minor early lifting that quickly stabilizes, and the lip corners show a slight outward pull initially that diminishes over time. The chin displays gradual wrinkling, accompanied by subtle upward pressing of the lower lip that fades gently. The lips remain unpressed and slightly parted at times, while the jaw lowers moderately and stabilizes. Overall, the facial movements flow smoothly, reflecting a coherent progression of expression without abrupt changes or interruptions.
"""


EXAMPLES_DICT = {
    "q3": {
        "description1": """
From frames 16 to 22, the cheeks display a moderate lift that gradually diminishes over time, establishing a subtle downward trend in activity. The corners of the lips initially curve upward slightly but steadily relax and straighten toward the end of the sequence. The chin exhibits a consistent forward push with some deepening of wrinkles, maintaining steady tension throughout. Around the mouth, the lips are noticeably pressed together with significant tension early on, which gradually fades as the pressing softens. No apparent lip parting occurs during these frames. The eyes remain partially closed, with a progressive widening evident toward the latter frames.
""",
        "description2": """
From frames 54 to 60, the cheeks stay slightly raised, maintaining a subtle upward tension throughout the sequence. The corners of the lips initially show faint upward movement but gradually relax and diminish over time. The chin consistently shows minor forward pushing with a steady deepening of tension under the mouth. As the sequence progresses, the lips experience mild outward stretching and pressing, which subtly increase before stabilizing. The lips start slightly parted, then progressively close and remain so by the end. The eyes remain open with no significant narrowing or shutting action observed.
""",
        "output": """
From frames 16 to 60, the facial expression evolves through gentle shifts in the cheeks, lips, and chin. Initially, the cheeks show a moderate lift that gradually softens, creating a subtle downward trend in elevation, while the corners of the lips curve upward slightly before relaxing and straightening over time. The chin maintains a consistent forward push with noticeable wrinkling that remains steady throughout the sequence. Around the mouth, the lips transition from firm pressing with early tension to a more relaxed state, occasionally stretching outward, and move from a slightly parted position to fully closed by the later frames. The eyes progress from partial closure toward a more open state, with no pronounced narrowing or blinking, resulting in an overall sequence that reflects smoothly changing facial dynamics with modest lip and cheek activity and steady chin tension.
""",
    },
    "q4": {
        "description1": """
From frames 64 to 70, the upper eyelids remain mostly steady with a very slight initial lift that dissipates toward the end. The area around the cheeks shows mild activation that diminishes mid-sequence before increasing slightly in the final frames. The lower part of the face is characterized by pronounced chin activity, starting with stronger prominence, which progressively decreases over time. The corners of the lips display minimal engagement with a fleeting upward motion that is barely perceptible and fades almost immediately. Overall, the movements suggest subtle tension around the chin and cheeks with very limited engagement in other areas of the face.
""",
        "description2": """
From frames 97 to 103, the eyebrows exhibit a subtle upward widening movement, with the intensity fluctuating slightly but showing an overall progressive lift toward the later frames. The upper eyelids display consistent tension, maintaining a moderate level of openness throughout, while the cheeks reveal faint activation, peaking slightly in the middle frames before relaxing. Toward the end of the sequence, there is an increasing downward pull at the chin area, accompanied by a mild tightening under the jaw. Throughout these frames, the activity in the facial muscles remains understated and focused on subtle adjustments around the eyes, brows, and lower face.
""",
        "output": """
From frames 64 to 103, the expression develops through a series of understated shifts across the eyes, cheeks, and lower face. Early in the sequence, the upper eyelids show only a faint lift that soon fades, while the cheeks reveal mild activity that dips before gently reemerging. The chin begins with noticeable prominence and gradually eases, reducing the sense of tension in the lower face. As the sequence progresses, the brows take on a subtle widening movement that becomes more apparent toward the later frames, accompanied by sustained openness in the eyelids. The cheeks briefly regain a touch of activation mid-sequence before softening again, while the chin shifts toward a downward pull and light tightening near the jaw. Taken together, these changes form a smooth progression marked by modest eye and brow adjustments, fluctuating cheek activity, and evolving but restrained tension in the chin and jaw.
""",
    },
    "q5": {
        "description1": """
From frames 148 to 154, the corners of the lips display a consistent upward lift that intensifies slightly toward the final frames, accompanied by a noticeable widening of the mouth opening that becomes more pronounced over time. The cheeks show moderate raising throughout, with the intensity gradually increasing to deepen the lines beneath the eyes. The lips remain visibly parted, with the degree of parting expanding steadily, while the upper lip rises mildly, contributing to greater tooth exposure. No significant horizontal stretching or pressing is observed, and the overall expression reflects a progressive increase in mouth opening and lip elevation supported by sustained cheek activation.
""",
        "description2": """
From frames 188 to 194, the corners of the lips progressively draw upward in a pronounced motion, peaking toward the middle and holding steady, before slightly easing toward the final frames. Simultaneously, the lips part noticeably wide at the beginning and maintain this openness with further subtle adjustments in the final moments. A gentle outward stretching of the lips becomes apparent midway but then gradually tapers off. The cheeks show lightening activity early on, which grows subtly stronger over time. There is an understated and consistent slackening of the lower lip area, giving way to visible moments of light tension, while the eyelids exhibit a mildly engaged, tightening appearance that amplifies slightly in the latter frames. The sequence highlights a dynamic interplay between upward lip movement and slight cheek activation, paired with a consistent modulation of lip parting and stretch.
""",
        "output": """
From frames 148 to 194, the facial expression evolves with a smooth interplay between the lips and cheeks. The corners of the lips rise steadily, reaching a pronounced lift mid-sequence before easing slightly toward the end, while the mouth remains widely parted throughout, peaking in openness near the middle and subtly adjusting thereafter. The cheeks display moderate activation that intensifies gradually before softening slightly in the latter frames. Alongside this, the lower lip shows moments of light tension interspersed with brief relaxation, and the lips exhibit a gentle outward stretch that emerges mid-sequence before diminishing. The eyelids remain subtly engaged, contributing to an overall dynamic that reflects a coordinated progression of lip elevation, mouth opening, and cheek activity, forming a cohesive and smoothly changing facial movement pattern.
""",
    },
    "q6": {
        "description1": """
From frames 138 to 144, the corners of the lips pull upward prominently, with the upward action gradually intensifying and reaching its peak toward the end of the sequence. The upper lip raises consistently over the duration, with a subtle but steady increase in visibility. The lips part slightly at first, with the separation becoming more pronounced as the sequence progresses, particularly in the latter frames, though the overall opening remains modest. Minimal tension is observed around the mouth throughout, and no horizontal stretching or pressing is noticeable. The overall movement highlights a progressive lift of the mouth corners and upper lip, accompanied by a controlled increase in mouth opening.
""",
        "description2": """
From frames 179 to 185, the corners of the lips progressively lift into an increasingly pronounced upward pull, becoming more prominent toward the end of the sequence. Concurrently, the upper lip raises gradually, with the action intensifying as the sequence progresses, leading to a more noticeable exposure of the upper teeth. The lips part at the beginning with a moderate opening that decreases slightly in the latter frames. Meanwhile, any visible tension or compression around the mouth seen early in the sequence softens and eventually disappears by the final frames, resulting in a relaxed configuration overall.
""",
        "output": """
From frames 138 to 185, the expression develops through a steady rise in the corners of the lips and an accompanying lift of the upper lip. Early on, the lips begin with only slight separation that gradually widens, while the upward pull of the mouth corners and the raising of the upper lip become increasingly pronounced, reaching their strongest expression mid-sequence. Toward the later frames, the mouth remains visibly lifted but with the opening reducing somewhat, shifting into a more contained shape. Any fleeting signs of tension around the lips present at the start fade away, leaving a relaxed overall configuration that highlights the continuity of lip elevation balanced by subtle modulation in parting.
""",
    },
}


def build_openai_client(OpenAI, api_key: str = "", base_url: str = ""):
    """
    Keep behavior compatible with the original scripts:
    - default api_key/base_url are empty strings (user fills them in)
    - but allow env var fallback for open-source friendliness
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    base_url = base_url or os.getenv("OPENAI_BASE_URL", "")
    return OpenAI(api_key=api_key, base_url=base_url)


def build_merge_prompt(description1: str, description2: str, example: str) -> str:
    # Intentionally matches the original prompt wording/structure.
    return f"""You are assisting in analyzing Facial Action Unit (AU) trends from videos. The data is segmented into small, non-continuous 7-frame windows centered around key frames. Each window has a pre-generated description of AU changes during that short span.

Your task is to integrate the AU descriptions from two or more of these small windows into a single, cohesive description that approximates the trend over a larger time range. These windows may have gaps between them, but your summary should ignore the missing frames and treat the group of small windows as a semantic unit.

Instructions:

  - Do NOT concatenate or copy-paste them.
  
  - Synthesize a new description that:

    - Reflects the overall change trend across the AU events.
    
    - The output must be a single, unified description.

    - Avoids redundant repetitions and a list or side-by-side summary.

    - Presents the change as a smooth progression.

    - Uses the merged frame range (e.g., “From frames 180 to 196”) even if frames between the windows (like 187–189) are not observed.

    - Focuses on expressing temporal flow and facial dynamics, even if intermediate frames are not present.

    - You may paraphrase and restructure as needed to ensure clarity and coherence.

--- 
Here is an example:
{example}

Now, using the example as a guide, synthesize the output as instructed:
Input:
description1:
{description1}

description2:
{description2}

Output:"""


def get_example_for_q(mode: str, q: str) -> str:
    if mode == "all":
        return EXAMPLE_ALL
    if mode != "selected":
        raise ValueError(f"Unknown mode '{mode}'. Expected: selected, all")

    example = EXAMPLES_DICT.get(q)
    if example is None:
        return "Example not found for the given q"
    return (
        "Input:\n"
        "description1:\n"
        f"{example['description1']}\n"
        "description2:\n"
        f"{example['description2']}\n"
        "Output:\n"
        f"{example['output']}"
    )


def make_prompt_for_mode(description1: str, description2: str, q: str, mode: str) -> str:
    example = get_example_for_q(mode=mode, q=q)
    return build_merge_prompt(description1, description2, example)


def make_llm_caller(client, model: str) -> Callable[[str], str]:
    def _call(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    return _call


def read_nonempty_lines(file_path: str) -> List[str]:
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def merge_descriptions(
    lines: Iterable[str],
    make_prompt: Callable[[str, str, str], str],
    llm_call: Callable[[str], str],
    q: Optional[str] = None,
) -> str:
    merged_so_far = ""
    q = q or ""

    for next_desc in lines:
        if merged_so_far == "":
            merged_so_far = next_desc
            continue
        prompt = make_prompt(merged_so_far, next_desc, q)
        merged_so_far = llm_call(prompt)

    return merged_so_far


def process_one_file(
    file_path: str,
    output_path: str,
    make_prompt: Callable[[str, str, str], str],
    llm_call: Callable[[str], str],
    q: Optional[str] = None,
) -> None:
    lines = read_nonempty_lines(file_path)
    merged = merge_descriptions(lines, make_prompt=make_prompt, llm_call=llm_call, q=q)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(merged)


def process_tree(
    description_root: str,
    output_root: str,
    q_list: List[str],
    split_list: List[str],
    make_prompt: Callable[[str, str, str], str],
    llm_call: Callable[[str], str],
    tqdm,
) -> None:
    for q in q_list:
        for split in split_list:
            in_dir = os.path.join(description_root, q, split)
            out_dir = os.path.join(output_root, q, split)
            os.makedirs(out_dir, exist_ok=True)

            for fname in tqdm(os.listdir(in_dir), desc=f"Processing {q}-{split}"):
                if not fname.endswith(".txt"):
                    continue
                in_path = os.path.join(in_dir, fname)
                out_path = os.path.join(out_dir, fname)

                if os.path.exists(out_path):
                    continue

                process_one_file(in_path, out_path, make_prompt=make_prompt, llm_call=llm_call, q=q)

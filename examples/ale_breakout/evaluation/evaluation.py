import numpy as np
import matplotlib.pyplot as plt


class Evaluation:
    """Thorough evaluation of a Breakout heuristic over many episodes.

    Uses a fixed seed per episode so results are fully reproducible and
    comparable across different heuristics.

    Args:
        n_episodes: number of complete Breakout games to play.
        max_steps:  step cap per episode.
    """

    # All 4 actions are valid; anything outside → NOOP
    _ACTION_MAP = {0: 0, 1: 1, 2: 2, 3: 3}

    def __init__(self, n_episodes: int = 20, max_steps: int = 5000):
        try:
            import ale_py          # noqa: F401  registers ALE envs with gymnasium
            import gymnasium as gym  # noqa: F401
        except ImportError:
            raise ImportError(
                'gymnasium and ale-py are required. Install with: '
                'pip install "gymnasium[atari]" ale-py && '
                'pip install autorom && AutoROM --accept-license'
            )
        self.n_episodes = n_episodes
        self.max_steps = max_steps

    def evaluate(self, heuristic_fn) -> dict:
        """Run heuristic_fn for n_episodes and return a stats dictionary.

        Args:
            heuristic_fn: callable(obs: np.ndarray) -> int

        Returns:
            dict with keys:
                scores   – list of per-episode total scores
                mean     – mean episode score
                std      – standard deviation
                best     – highest single-episode score
                worst    – lowest single-episode score
        """
        import ale_py          # noqa: F401
        import gymnasium as gym

        env = gym.make('ALE/Breakout-v5', obs_type='ram', render_mode=None)
        scores = []
        try:
            for ep in range(self.n_episodes):
                obs, _ = env.reset(seed=ep)
                episode_score = 0.0
                for _ in range(self.max_steps):
                    action = self._ACTION_MAP.get(int(heuristic_fn(obs)), 0)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    episode_score += reward
                    if terminated or truncated:
                        break
                scores.append(episode_score)
        finally:
            env.close()

        scores_arr = np.array(scores)
        return {
            'scores': scores,
            'mean':   float(np.mean(scores_arr)),
            'std':    float(np.std(scores_arr)),
            'best':   float(np.max(scores_arr)),
            'worst':  float(np.min(scores_arr)),
        }

    def plot_scores(
        self,
        scores: list,
        label: str = 'heuristic',
        save_path: str = 'scores.png',
    ) -> None:
        """Bar chart of per-episode scores with mean line and std band.

        Args:
            scores:    list of episode scores (from evaluate()).
            label:     heuristic name shown in the plot title.
            save_path: output filename (.png); a matching .pdf is also saved.
        """
        scores_arr = np.array(scores)
        mean = scores_arr.mean()
        std  = scores_arr.std()
        ep_idx = np.arange(1, len(scores) + 1)

        fig, ax = plt.subplots(figsize=(max(8, len(scores) * 0.45), 5))
        ax.bar(ep_idx, scores_arr, color='#3498db', edgecolor='white',
               linewidth=0.5, zorder=2, label='episode score')
        ax.axhline(mean, color='#2c3e50', linewidth=1.8, linestyle='--',
                   label=f'mean = {mean:.1f}', zorder=3)
        ax.axhspan(mean - std, mean + std, alpha=0.12, color='#2c3e50',
                   label=f'±1 std = {std:.1f}', zorder=1)

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Score (bricks × pts)', fontsize=12)
        ax.set_title(
            f'Breakout episode scores — {label}  (n = {len(scores)})',
            fontsize=13,
        )
        ax.set_xticks(ep_idx)
        ax.set_xlim(0.3, len(scores) + 0.7)
        ax.set_ylim(0, max(scores_arr.max() * 1.15, 10))
        ax.legend(fontsize=11)
        ax.grid(axis='y', linestyle=':', alpha=0.4, zorder=0)

        ax.text(0.985, 0.97, f'best {int(scores_arr.max())} / worst {int(scores_arr.min())}',
                transform=ax.transAxes, ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#bdc3c7', alpha=0.9))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.savefig(save_path.rsplit('.', 1)[0] + '.pdf')
        plt.close(fig)
        print(f'  → score plot  : {save_path}')

    def render_episode(
        self,
        heuristic_fn,
        save_path: str = 'gameplay.gif',
        seed: int = 100,
        render_max_steps: int = 2000,
        frame_skip: int = 4,
        scale: int = 3,
        duration: int = 100,
    ) -> float:
        """Play one episode and save an enlarged, annotated animated GIF.

        Args:
            heuristic_fn:     callable(obs: np.ndarray) -> int
            save_path:        output .gif filename.
            seed:             env seed for reproducibility.
            render_max_steps: step cap for the rendered episode.
            frame_skip:       save one frame every this many gymnasium steps.
            scale:            integer upscale factor (default 3).
            duration:         ms per GIF frame (default 100 ≈ 10 fps).

        Returns:
            Total episode score.

        Requires: Pillow  (pip install pillow)
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            print('  Pillow not found — skipping GIF (pip install pillow)')
            return 0.0

        import ale_py          # noqa: F401
        import gymnasium as gym

        # ── colour palette ────────────────────────────────────────────────
        BG_PANEL   = ( 22,  22,  38)
        BG_SIDEBAR = ( 30,  30,  52)
        DIVIDER    = ( 75,  75, 120)
        CLR_TEXT   = (200, 200, 215)
        CLR_HEAD   = (255, 255, 255)
        CLR_BAR_FG = ( 52, 152, 219)
        CLR_BAR_BG = ( 44,  62,  80)

        ACTION_INFO = {
            0: ('NOOP',  (149, 165, 166)),
            1: ('FIRE',  (241, 196,  15)),
            2: ('RIGHT', ( 52, 152, 219)),
            3: ('LEFT',  (230, 126,  34)),
        }

        SIDEBAR_W = 200
        PAD       = 12

        def _font(size):
            try:
                return ImageFont.load_default(size=size)
            except TypeError:
                return ImageFont.load_default()

        font_sm = _font(11)
        font_md = _font(13)
        font_lg = _font(17)

        env = gym.make('ALE/Breakout-v5', obs_type='ram', render_mode='rgb_array')
        obs, _ = env.reset(seed=seed)

        frames       = []
        total_score  = 0.0
        step         = 0
        last_action  = 0
        terminated   = truncated = False

        try:
            while not (terminated or truncated) and step < render_max_steps:
                last_action = self._ACTION_MAP.get(int(heuristic_fn(obs)), 0)
                obs, reward, terminated, truncated, _ = env.step(last_action)
                total_score += reward
                step        += 1

                if step % frame_skip != 0:
                    continue

                # ── scale game image ──────────────────────────────────────
                pixel_frame = env.render()
                orig_h, orig_w = pixel_frame.shape[:2]
                game_w = orig_w * scale
                game_h = orig_h * scale

                canvas = Image.new('RGB', (game_w + SIDEBAR_W, game_h), BG_PANEL)
                canvas.paste(
                    Image.fromarray(pixel_frame).resize(
                        (game_w, game_h), Image.NEAREST
                    ),
                    (0, 0),
                )

                # ── sidebar ───────────────────────────────────────────────
                draw = ImageDraw.Draw(canvas)
                draw.rectangle([game_w, 0, game_w + SIDEBAR_W - 1, game_h - 1],
                               fill=BG_SIDEBAR)
                draw.line([game_w, 0, game_w, game_h], fill=DIVIDER, width=2)

                sx, sy = game_w + PAD, PAD

                # title
                draw.text((sx, sy), 'EoH Breakout', font=font_md, fill=CLR_HEAD)
                sy += 19
                draw.line([sx, sy, game_w + SIDEBAR_W - PAD, sy],
                          fill=DIVIDER, width=1)
                sy += 10

                # action
                draw.text((sx, sy), 'ACTION', font=font_sm, fill=CLR_TEXT)
                sy += 15
                act_label, act_clr = ACTION_INFO.get(last_action, ('?', CLR_TEXT))
                draw.text((sx, sy), act_label, font=font_lg, fill=act_clr)
                ax = sx + 72
                if last_action == 2:    # RIGHT arrow →
                    draw.polygon(
                        [(ax, sy + 4), (ax, sy + 12), (ax + 12, sy + 8)],
                        fill=act_clr,
                    )
                elif last_action == 3:  # LEFT arrow ←
                    draw.polygon(
                        [(ax + 12, sy + 4), (ax + 12, sy + 12), (ax, sy + 8)],
                        fill=act_clr,
                    )
                elif last_action == 1:  # FIRE ★
                    draw.ellipse([ax + 2, sy + 4, ax + 10, sy + 12], fill=act_clr)
                else:                   # NOOP –
                    draw.rectangle([ax + 1, sy + 7, ax + 11, sy + 9], fill=act_clr)
                sy += 28

                # score
                draw.text((sx, sy), 'SCORE', font=font_sm, fill=CLR_TEXT)
                sy += 15
                draw.text((sx, sy), f'{total_score:.0f}', font=font_lg,
                          fill=CLR_HEAD)
                sy += 28

                # step + progress bar
                draw.text((sx, sy), 'STEP', font=font_sm, fill=CLR_TEXT)
                sy += 15
                draw.text((sx, sy), f'{step} / {render_max_steps}',
                          font=font_md, fill=CLR_TEXT)
                sy += 17
                bar_w  = SIDEBAR_W - 2 * PAD
                filled = max(1, int(bar_w * step / render_max_steps))
                draw.rectangle([sx, sy, sx + bar_w, sy + 7], fill=CLR_BAR_BG)
                draw.rectangle([sx, sy, sx + filled, sy + 7], fill=CLR_BAR_FG)
                sy += 21

                # divider
                draw.line([sx, sy, game_w + SIDEBAR_W - PAD, sy],
                          fill=DIVIDER, width=1)
                sy += 10

                # RAM state
                draw.text((sx, sy), 'RAM STATE', font=font_sm, fill=CLR_TEXT)
                sy += 15
                for lbl, val in (
                    ('Ball X',   int(obs[99])),
                    ('Ball Y',   int(obs[101])),
                    ('Paddle X', int(obs[72])),
                ):
                    draw.text((sx, sy), f'{lbl:<10}', font=font_sm, fill=CLR_TEXT)
                    draw.text((sx + 90, sy), f'{val:>3}', font=font_sm, fill=CLR_HEAD)
                    sy += 14

                frames.append(canvas)
        finally:
            env.close()

        if not frames:
            print('  No frames captured.')
            return total_score

        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )
        print(f'  → gameplay GIF: {save_path}  '
              f'({len(frames)} frames, {duration} ms/frame, '
              f'score {total_score:.0f})')
        return total_score

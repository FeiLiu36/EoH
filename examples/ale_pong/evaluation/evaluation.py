import numpy as np
import matplotlib.pyplot as plt


class Evaluation:
    """Thorough evaluation of a Pong heuristic over many episodes.

    Uses a fixed seed per episode so results are fully reproducible and
    comparable across different heuristics.

    Args:
        n_episodes: number of complete Pong games to play.
        max_steps:  step cap per episode (a full Pong game rarely exceeds 2000
                    steps with ALE's default 4-frame skip).
    """

    # actions 4/5 (UP+FIRE / DOWN+FIRE) → UP/DOWN; FIRE has no effect in Pong
    _ACTION_MAP = {0: 0, 1: 0, 2: 2, 3: 3, 4: 2, 5: 3}

    def __init__(self, n_episodes: int = 20, max_steps: int = 2000):
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
                scores      – list of per-episode total rewards
                mean        – mean episode reward
                std         – standard deviation of episode rewards
                win_rate    – fraction of episodes with reward > 0
                best        – highest single-episode reward
                worst       – lowest single-episode reward
        """
        import ale_py          # noqa: F401
        import gymnasium as gym

        env = gym.make('ALE/Pong-v5', obs_type='ram', render_mode=None)
        scores = []
        try:
            for ep in range(self.n_episodes):
                obs, _ = env.reset(seed=ep)
                episode_reward = 0.0
                for _ in range(self.max_steps):
                    raw = heuristic_fn(obs)
                    action = self._ACTION_MAP.get(int(raw), 0)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    if terminated or truncated:
                        break
                scores.append(episode_reward)
        finally:
            env.close()

        scores_arr = np.array(scores)
        return {
            'scores':    scores,
            'mean':      float(np.mean(scores_arr)),
            'std':       float(np.std(scores_arr)),
            'win_rate':  float(np.mean(scores_arr > 0)),
            'best':      float(np.max(scores_arr)),
            'worst':     float(np.min(scores_arr)),
        }

    def plot_scores(
        self,
        scores: list,
        label: str = 'heuristic',
        save_path: str = 'scores.png',
    ) -> None:
        """Bar chart of per-episode scores with win/loss colouring and mean line.

        Args:
            scores:    list of episode rewards (as returned by evaluate()).
            label:     heuristic name shown in the plot title.
            save_path: output filename (.png); a matching .pdf is also saved.
        """
        scores_arr = np.array(scores)
        mean = scores_arr.mean()
        std  = scores_arr.std()
        ep_idx = np.arange(1, len(scores) + 1)

        colours = [
            '#2ecc71' if s > 0 else '#e74c3c' if s < 0 else '#95a5a6'
            for s in scores
        ]

        fig, ax = plt.subplots(figsize=(max(8, len(scores) * 0.45), 5))
        ax.bar(ep_idx, scores_arr, color=colours, edgecolor='white', linewidth=0.5,
               zorder=2)
        ax.axhline(mean, color='#2c3e50', linewidth=1.8, linestyle='--',
                   label=f'mean = {mean:+.2f}', zorder=3)
        ax.axhspan(mean - std, mean + std, alpha=0.12, color='#2c3e50',
                   label=f'±1 std = {std:.2f}', zorder=1)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.35,
                   zorder=2)

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Score (player − CPU)', fontsize=12)
        ax.set_title(f'Pong episode scores — {label}  (n = {len(scores)})',
                     fontsize=13)
        ax.set_xticks(ep_idx)
        ax.set_xlim(0.3, len(scores) + 0.7)
        ax.set_ylim(-23, 23)
        ax.legend(fontsize=11)
        ax.grid(axis='y', linestyle=':', alpha=0.4, zorder=0)

        wins   = int((scores_arr > 0).sum())
        losses = int((scores_arr < 0).sum())
        ax.text(0.985, 0.97, f'W {wins} / L {losses}',
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
        seed: int = 0,
        render_max_steps: int = 1500,
        frame_skip: int = 4,
        scale: int = 3,
        duration: int = 100,
    ) -> float:
        """Play one episode and save an enlarged, annotated animated GIF.

        The game image is scaled up (nearest-neighbour, preserves pixel art)
        and a sidebar panel is added showing the current action, cumulative
        score, step progress bar, and live RAM state values.

        Args:
            heuristic_fn:     callable(obs: np.ndarray) -> int
            save_path:        output .gif filename.
            seed:             env seed for reproducibility.
            render_max_steps: step cap for the rendered episode.
            frame_skip:       save one frame every this many gymnasium steps.
            scale:            integer upscale factor for the game image (default 3).
            duration:         ms per GIF frame — higher = slower (default 100).

        Returns:
            Total episode reward.

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
        CLR_UP     = ( 46, 204, 113)   # green
        CLR_DOWN   = (231,  76,  60)   # red
        CLR_NOOP   = (149, 165, 166)   # grey
        CLR_BAR_FG = ( 52, 152, 219)   # blue progress fill
        CLR_BAR_BG = ( 44,  62,  80)   # progress background

        ACTION_INFO = {
            0: ('NOOP', CLR_NOOP),
            2: ('UP',   CLR_UP),
            3: ('DOWN', CLR_DOWN),
            4: ('UP',   CLR_UP),
            5: ('DOWN', CLR_DOWN),
        }

        SIDEBAR_W = 200
        PAD       = 12

        # ── load fonts (Pillow >=10 supports size; older falls back) ──────
        def _font(size):
            try:
                return ImageFont.load_default(size=size)
            except TypeError:
                return ImageFont.load_default()

        font_sm = _font(11)
        font_md = _font(13)
        font_lg = _font(17)

        env = gym.make('ALE/Pong-v5', obs_type='ram', render_mode='rgb_array')
        obs, _ = env.reset(seed=seed)

        frames      = []
        total_reward = 0.0
        step         = 0
        last_action  = 0
        terminated   = truncated = False

        try:
            while not (terminated or truncated) and step < render_max_steps:
                raw         = heuristic_fn(obs)
                last_action = self._ACTION_MAP.get(int(raw), 0)
                obs, reward, terminated, truncated, _ = env.step(last_action)
                total_reward += reward
                step         += 1

                if step % frame_skip != 0:
                    continue

                # ── scale game image ──────────────────────────────────────
                pixel_frame = env.render()                    # (H, W, 3)
                orig_h, orig_w = pixel_frame.shape[:2]
                game_w = orig_w * scale
                game_h = orig_h * scale

                canvas_w = game_w + SIDEBAR_W
                canvas_h = game_h
                canvas = Image.new('RGB', (canvas_w, canvas_h), BG_PANEL)

                game_img = Image.fromarray(pixel_frame).resize(
                    (game_w, game_h), Image.NEAREST
                )
                canvas.paste(game_img, (0, 0))

                # ── sidebar background + dividing line ────────────────────
                draw = ImageDraw.Draw(canvas)
                draw.rectangle(
                    [game_w, 0, canvas_w - 1, canvas_h - 1],
                    fill=BG_SIDEBAR,
                )
                draw.line([game_w, 0, game_w, canvas_h],
                          fill=DIVIDER, width=2)

                sx = game_w + PAD   # sidebar left edge (x)
                sy = PAD            # running y cursor

                # ── title ─────────────────────────────────────────────────
                draw.text((sx, sy), 'EoH Pong Agent',
                          font=font_md, fill=CLR_HEAD)
                sy += 19
                draw.line([sx, sy, canvas_w - PAD, sy],
                          fill=DIVIDER, width=1)
                sy += 10

                # ── action ────────────────────────────────────────────────
                draw.text((sx, sy), 'ACTION', font=font_sm, fill=CLR_TEXT)
                sy += 15
                act_label, act_clr = ACTION_INFO.get(last_action,
                                                      ('?', CLR_NOOP))
                draw.text((sx, sy), act_label, font=font_lg, fill=act_clr)
                # draw a filled triangle arrow next to the label
                ax = sx + 62
                if last_action in (2, 4):   # UP – triangle pointing up
                    draw.polygon(
                        [(ax, sy + 14), (ax + 12, sy + 14), (ax + 6, sy + 2)],
                        fill=act_clr,
                    )
                elif last_action in (3, 5): # DOWN – triangle pointing down
                    draw.polygon(
                        [(ax, sy + 2), (ax + 12, sy + 2), (ax + 6, sy + 14)],
                        fill=act_clr,
                    )
                else:                        # NOOP – horizontal bar
                    draw.rectangle(
                        [ax + 1, sy + 7, ax + 11, sy + 9],
                        fill=act_clr,
                    )
                sy += 28

                # ── score ─────────────────────────────────────────────────
                draw.text((sx, sy), 'SCORE', font=font_sm, fill=CLR_TEXT)
                sy += 15
                score_clr = (CLR_UP   if total_reward > 0 else
                             CLR_DOWN if total_reward < 0 else CLR_NOOP)
                draw.text((sx, sy), f'{total_reward:+.0f}',
                          font=font_lg, fill=score_clr)
                sy += 28

                # ── step + progress bar ───────────────────────────────────
                draw.text((sx, sy), 'STEP', font=font_sm, fill=CLR_TEXT)
                sy += 15
                draw.text((sx, sy), f'{step} / {render_max_steps}',
                          font=font_md, fill=CLR_TEXT)
                sy += 17
                bar_w  = SIDEBAR_W - 2 * PAD
                bar_h  = 7
                filled = max(1, int(bar_w * step / render_max_steps))
                draw.rectangle([sx, sy, sx + bar_w, sy + bar_h],
                               fill=CLR_BAR_BG)
                draw.rectangle([sx, sy, sx + filled, sy + bar_h],
                               fill=CLR_BAR_FG)
                sy += bar_h + 14

                # ── divider ───────────────────────────────────────────────
                draw.line([sx, sy, canvas_w - PAD, sy],
                          fill=DIVIDER, width=1)
                sy += 10

                # ── RAM state ─────────────────────────────────────────────
                draw.text((sx, sy), 'RAM STATE', font=font_sm, fill=CLR_TEXT)
                sy += 15
                for lbl, val in (
                    ('Ball X',    int(obs[49])),
                    ('Ball Y',    int(obs[54])),
                    ('Player Y',  int(obs[51])),
                    ('CPU Y',     int(obs[50])),
                ):
                    draw.text((sx, sy), f'{lbl:<10}', font=font_sm,
                              fill=CLR_TEXT)
                    draw.text((sx + 90, sy), f'{val:>3}', font=font_sm,
                              fill=CLR_HEAD)
                    sy += 14

                frames.append(canvas)
        finally:
            env.close()

        if not frames:
            print('  No frames captured.')
            return total_reward

        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )
        print(f'  → gameplay GIF: {save_path}  '
              f'({len(frames)} frames, {duration} ms/frame, '
              f'score {total_reward:+.0f})')
        return total_reward

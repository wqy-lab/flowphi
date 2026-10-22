from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

from flowphi.ui.model import PlanDocument, read_document, write_json_atomic
from flowphi.ui.sources import DEFAULT_CACHE, ROOT, file_digest, load_source
from flowphi.config import PACKAGE
from flowphi.progress import report_log, report_progress
from flowphi.runtime import NO_WINDOW, module_command


def event(message, **values):
    stage = values.pop('stage', message)
    report_progress(stage, message, **values)


def planning_progress(done, total, task):
    detail = f'当前音符 {task.id} · {task.beat / 1000:.3f} s' if task else '全部音符已处理'
    phrase = getattr(task, 'planning_detail', None)
    if phrase:
        names = dict(baseline='逐颗兜底', heads='夹键分配', continuous='连续划动', relay='分段接力', stationary='多指驻点', hand_motion='受限手部评估',hand_repair='手部失败修复',readability='手别与遮挡复核',ordinary_hand='普通音符手部筛选',ordinary_audit='普通音符手部复核')
        scope = '黄键长条' if phrase['strategy'] in ('baseline', 'heads', 'continuous', 'relay', 'stationary') else '动作窗口'
        detail = (f'{scope} {phrase["start"]:.3f}–{phrase["end"]:.3f} s · '
                  f'{names.get(phrase["strategy"], phrase["strategy"])} {phrase["done"]}/{phrase["total"]}')
        if phrase.get('elapsed_seconds') is not None:
            detail += f' · 已用 {phrase["elapsed_seconds"]:.1f} s'
    event('生成指法初稿', stage='plan', done=done, total=total, unit='个音符', detail=detail)


def run(command, *, cwd=ROOT, log=None):
    with subprocess.Popen(list(map(str, command)), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding='utf-8', errors='replace', creationflags=NO_WINDOW) as process:
        try:
            for line in process.stdout:
                if log:
                    log.write(line)
                    log.flush()
                print(line.rstrip(), flush=True)
            if process.wait():
                raise RuntimeError(f'子任务失败（退出码 {process.returncode}），详情见任务日志')
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def measure_profile(options, cache):
    from flowphi.rendering.pipeline import find_blender
    from flowphi.motion import default_profile
    if not options.get('model'):
        return default_profile()
    model = Path(options['model']).resolve()
    if not model.is_file():
        raise ValueError('找不到所选手模文件')
    blender = find_blender(options.get('blender') or None)
    digest = file_digest(model)
    scale = float(options.get('hand_scale', .27))
    key = hashlib.sha256((digest + str(scale) + file_digest(PACKAGE / 'rendering/blender.py') + str(Path(blender).stat().st_mtime_ns)).encode()).hexdigest()
    folder = cache / 'profiles' / key
    profile_path = folder / 'hand-profile.json'
    if not profile_path.is_file():
        event('测量手部模型', stage='profile', detail='同一模型下次会复用测量结果')
        folder.mkdir(parents=True, exist_ok=True)
        job = folder / 'profile-job.json'
        write_json_atomic(job, dict(output=str(folder), hand_scale=scale, enable_thumb=True, profile_only=True))
        with (folder / 'profile.log').open('w', encoding='utf-8') as log:
            run([blender, '--background', '--factory-startup', '--disable-autoexec', model,
                 '--python-exit-code', '1', '--python', PACKAGE / 'rendering/blender.py', '--', job], log=log)
    else:
        report_log('复用已测量的手型：' + str(profile_path))
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    profile['model_sha256'] = digest
    return profile


def load_inputs(request):
    document = read_document(request['project'])
    source = load_source(request.get('chart') or document.chart.get('path'), request.get('cache', DEFAULT_CACHE),
                         expected=document.chart, audio=document.chart.get('audio'), background=document.chart.get('background'))
    document.chart = source.reference
    return document, source


def validate_export(document, source, *, time_range=None):
    from flowphi.services.candidates import validate_compiled
    event('检查触控判定', stage='validate', detail='检查手指占用、判定区域和 Hold 连续性')
    return validate_compiled(document, source.chart, time_range=time_range,
        progress=lambda done, total: event('检查判定区域', stage='validate', done=done, total=total, unit='段动作'))


def execute(request):
    kind = request['kind']
    work = Path(request['work']).resolve()
    work.mkdir(parents=True, exist_ok=True)
    cache = Path(request.get('cache', DEFAULT_CACHE)).resolve()
    if kind == 'model_python':
        from flowphi.ui.model_python import calculate
        return calculate(request['code'], request['data'])
    event('读取谱面与工程', stage='input')
    if kind == 'generate':
        from flowphi.planning.planner import Settings, plan
        from rich.console import Console
        reference = request.get('chart_reference', {})
        source = load_source(request['chart'], cache, expected=reference,
                             audio=reference.get('audio'), background=reference.get('background'))
        options = request.get('options', {})
        settings = Settings(**options.get('settings', {}))
        profile = measure_profile(options, cache)
        report_log(f"谱面：{source.reference['path']} · {source.reference['note_count']} 个音符")
        if source.reference.get('chart_entry'):
            report_log('包内谱面：' + source.reference['chart_entry'])
        report_log(f"分配设置：{len(settings.fingers) * 2} 根手指，每步保留 {settings.beam_width} 个候选")
        event('生成指法初稿', stage='plan')
        console = Console(force_terminal=False, color_system=None)
        def checkpoint(motion):
            document = PlanDocument(motion, chart=source.reference, pose=options.get('pose'),
                                    title=Path(request['chart']).stem + ' (recovery draft)', copy_baseline=False)
            write_json_atomic(work / 'recovery.fingering.json', document.snapshot())
        if options.get('keep_unplanned', True):
            from flowphi.ui.generation import generate_draft
            motion = generate_draft(source.chart, settings, profile, console, progress=planning_progress,
                                    checkpoint=checkpoint)
        else:
            _, _, motion = plan(source.chart, console=console, settings=settings, profile=profile, progress=planning_progress)
        document = PlanDocument(motion, chart=source.reference, pose=options.get('pose'),
                                title=Path(request['chart']).stem, copy_baseline=False)
        project = work / 'generated.fingering.json'
        event('保存指法初稿', stage='save')
        write_json_atomic(project, document.snapshot())
        pending = motion['diagnostics'].get('draft_fallback_notes', [])
        if pending:
            event(f'初稿已保留全部音符；其中 {len(pending)} 个音符需要人工检查，可在“问题”页定位')
        hand_failed = motion['diagnostics'].get('ribbons', {}).get('hand_motion_failed_windows', 0)
        hand_failed += motion['diagnostics'].get('ordinary_hand_search',{}).get('final_failed_windows',0)
        if hand_failed:
            event(f'其中 {hand_failed} 段黄键长条未通过受限手模型检查，已保留草稿并在“问题”页标记')
        return dict(project=str(project), contacts=len(document), chart=source.reference)
    document, source = load_inputs(request)
    report_log(f"谱面：{source.reference['path']} · {source.reference['note_count']} 个音符 · {len(document)} 段动作")
    if kind == 'evaluate_candidate':
        from flowphi.services.candidates import evaluate_candidate
        event('评估候选打法', stage='candidate', detail='在副本上试改，完整检查原方案与候选方案')
        return evaluate_candidate(document, source.chart, request.get('operations', []),
            offset=request.get('offset', 0.), selected_ids=request.get('selected_ids', []),
            fallback_range=request.get('fallback_range'),
            progress=lambda done, total: event('检查候选判定区域', stage='candidate.validate',
                                               done=done, total=total, unit='段动作'))
    if kind == 'replan':
        from flowphi.ui.optimise import reassign
        event('重新分配手指', stage='replan', detail=f"已选 {len(request['selected'])} 段动作；跳过已锁定的指法")
        return reassign(document, request['selected'], width=request.get('beam_width', 32),
                        max_contacts=request.get('max_contacts', 1200), chart=source.chart,
                        hands=request.get('hands'), fingers=request.get('fingers'))
    if kind == 'zone':
        from flowphi.planning.contacts import ContactGeometry
        from flowphi.ui.contact_state import resolve_contacts
        contacts, groups, _ = resolve_contacts(document, source.chart)
        c = contacts[groups[request['contact']]]
        t = float(request['time'])
        g = ContactGeometry(source.chart, physical_screen=document.baseline['physical_screen'],
                            extra_clearance_m=document.baseline.get('comfort', {}).get('extra_block_clearance_m', 0.))
        area = g.zone(c, t)
        components = [area] if area.geom_type == 'Polygon' else getattr(area, 'geoms', [])
        polygons = [dict(exterior=list(p.exterior.coords), holes=[list(r.coords) for r in p.interiors])
                    for p in components if p.geom_type == 'Polygon']
        return dict(polygons=polygons, contact=request['contact'], time=t)
    if kind == 'validate':
        from flowphi.ui.validation import validate
        event('检查触控判定', stage='validate')
        return validate(document, source.chart, progress=lambda done, total: event('检查判定区域', stage='validate', done=done, total=total, unit='段动作'))
    if kind not in ('export', 'render'):
        raise ValueError('未知的编辑器任务')
    if kind == 'render':
        from flowphi.rendering.pipeline import find_blender
        options = request.get('options', {})
        background = options.get('background') or source.reference.get('background')
        if background:
            background = Path(background).resolve()
            if not background.is_file():
                raise ValueError('找不到关联曲绘，请重新选择曲绘文件')
            source.reference['background'] = str(background)
            document.chart = dict(source.reference)
            report_log('曲绘：' + str(background))
        else:
            report_log('未关联曲绘，本次视频将使用纯色背景可通过“文件 → 关联曲绘”选择图片', 'warning')
        report_log(f"视频设置：{options.get('width', 1280)}×{options.get('height', 720)} · {options.get('fps', 30)} FPS")
        report_log('手部模型：' + str(options.get('model', '')))
        report_log('游戏资源：' + str(options.get('resources', '')))
        model = Path(options.get('model', ''))
        if not model.is_file():
            raise ValueError('请选择用于三维渲染的手模 .blend 文件')
        if not source.reference.get('audio') or not Path(source.reference['audio']).is_file():
            raise ValueError('谱面未关联可用音乐，请先选择音频文件')
        if not options.get('resources') or not Path(options['resources']).exists():
            raise ValueError('请选择本地素材包 ZIP 或资源目录')
        find_blender(options.get('blender') or None)
        expected_model = document.baseline['profile'].get('model_sha256')
        if expected_model and file_digest(model) != expected_model:
            raise ValueError('所选手模与生成此方案时的模型不一致，请选择原手模')
    output = Path(request['output']).resolve()
    if any((output / name).exists() for name in ('plan.psap', 'motion-plan.json')):
        raise ValueError('目标目录已有导出的动作数据，请选择新的目录')
    time_range = None
    if kind == 'render' and not options.get('full'):
        start, duration = float(options.get('start', 0.)), float(options.get('duration', 5.))
        fps = int(options.get('fps', 30))
        if not math.isfinite(start) or not math.isfinite(duration) or start < 0 or duration <= 0 or fps <= 0:
            raise ValueError('片段的开始时间、时长或帧率无效')
        # Rendering rounds the last frame up. Touches stay on their millisecond
        # timebase; carry-in Holds retain their real DOWN for head validation.
        low = math.floor((start - source.chart.offset) * 1000) / 1000
        high = math.ceil((start + math.ceil(duration * fps) / fps - source.chart.offset) * 1000) / 1000
        time_range = [low, high]
        event(f'导出片段 {start:.3f}–{start + duration:.3f} s 内的动作')
    report = (validate_export(document, source, time_range=time_range) if request.get('auto_check', False)
              else dict(valid=None, issues=[], geometry_checked=False, skipped=True, time_range=time_range))
    if not report.get('skipped') and not report['valid']:
        return dict(validation=report, exported=False)
    from flowphi.ui.compiler import export_plan
    event('写入动作数据', stage='export', detail=str(output))
    export_plan(document, source, output, report, time_range=time_range)
    if kind == 'export':
        return dict(output=str(output), validation=report, exported=True)
    from flowphi.rendering.pipeline import find_blender
    options = request.get('options', {})
    model = Path(options.get('model', ''))
    if not model.is_file():
        raise ValueError('请选择用于三维渲染的手模 .blend 文件')
    if not source.reference.get('audio'):
        raise ValueError('谱面未关联音乐，请先选择音频文件')
    resources = Path(options.get('resources', ''))
    if not resources.exists():
        raise ValueError('请选择本地素材包 ZIP 或资源目录')
    blender = find_blender(options.get('blender') or None)
    render_output = output / 'render'
    command = module_command('flowphi.rendering.pipeline', source.reference['path'],
               '--model', model, '--resources', resources, '--blender', blender,
               '--psap', output / 'plan.psap', '--motion-plan', output / 'motion-plan.json',
               '--audio', source.reference['audio'], '--stream',
               '--hand-scale', str(document.baseline['profile']['scale']),
               '--width', str(options.get('width', 1280)), '--height', str(options.get('height', 720)),
               '--fps', str(options.get('fps', 30)), '--output', render_output)
    if source.reference.get('chart_entry'):
        command += ['--chart-entry', source.reference['chart_entry']]
    if source.reference.get('background'):
        command += ['--background', source.reference['background']]
    if options.get('full'):
        command += ['--full']
    else:
        command += ['--start', str(options.get('start', 0.)), '--duration', str(options.get('duration', 5.))]
    event('准备视频渲染', stage='render.prepare', detail='生成期间可继续编辑，也可取消本次任务')
    with (work / 'render-process.log').open('w', encoding='utf-8') as log:
        run(command, log=log)
    video = render_output / 'handcam.mp4'
    if not video.is_file():
        raise RuntimeError('渲染进程结束，但未找到输出视频')
    return dict(output=str(output), video=str(video), validation=report, exported=True)


def main():
    os.environ.setdefault('FLOWPHI_PROGRESS_JSON', '1')
    request = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    result_path = Path(request['work']) / 'result.json'
    try:
        report_log('任务类型：' + request['kind'])
        report_log('任务日志：' + str(Path(request['work']) / 'job.log'))
        if request.get('output'):
            report_log('输出目录：' + request['output'])
        result = execute(request)
        write_json_atomic(result_path, dict(ok=True, result=result))
        event('任务完成', stage='complete', done=1, total=1)
        return 0
    except Exception as exc:
        traceback.print_exc()
        error = str(exc)
        recovery = Path(request['work']) / 'recovery.fingering.json'
        result = dict(ok=False, error=error)
        if request.get('kind') == 'generate' and recovery.is_file():
            result['recovery_project'] = str(recovery)
            error += '\n已保存恢复初稿（最终复核未完成）：' + str(recovery)
            result['error'] = error
        write_json_atomic(result_path, result)
        report_log(error, 'error')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

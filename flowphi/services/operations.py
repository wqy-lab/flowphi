"""Window-independent, atomic application of AI editor operations."""
from __future__ import annotations
import copy
import uuid
from flowphi.pose import TARGETS, validate_values
from flowphi.ui.model import PlanDocument, finite


OPERATION_PARAMETERS = {
    'assign': '{hand: left|right, finger: index|middle|ring|little|thumb}；同时锁定指法',
    'alternate': '{finger, first: left|right}；按时间左右交替',
    'swap': '{}；交换左右手',
    'lock': '{locked: bool, path: bool=false}；锁定指法或轨迹',
    'paths': '{operation: translate|set|freeze|straighten|simplify|delete, x?: number, y?: number, '
             'rows?: 零起始控制点编号数组, only_hold?: bool, tolerance?: 米}；未传 rows 修改全部控制点；freeze 固定到所选首点，指定新位置用 set，x/y 仅用于 translate/set',
    'release': '{mode: offset|duration|absolute, value: 秒, reshape: trim|scale}；提前 offset 用负数',
    'flick': '{direction?: up|down|left|right|custom, angle?: 度, angle_space?: line|screen, '
             'length?: 屏幕宽度比例, duration_ms?: 毫秒, lead_ms?: 提前毫秒, release_ms?: 停留毫秒, note_id?: 音符编号}；未提供的参数沿用 Flick 对话框推算的当前值',
    'touch': '{mode: auto|continue|press}',
    'contact': '{start?: 秒, end?: 秒, points?: [[音乐秒,x,y],...], hand?, finger?, assignment_locked?, path_locked?, touch_mode?}；points 首点时间=start，严格递增且均<end，末点位置保持到抬起',
    'pose_defaults': '{hand: left|right, target: hand|index|middle|ring|little|thumb, values: {姿态通道: 数值}}；合并默认姿态',
    'pose_clip': '完整片段 {id?: 更新已有片段的id, hand, target, start, end, fade_in?, fade_out?, '
                 'curve?: smooth|linear, enabled?: bool, keys: [{time: 音乐秒, values: {姿态通道: 数值}}]}；smooth 为五次缓动，linear 为匀速；淡入淡出始终平滑，结束回到其他片段与默认值合成的姿态',
    'remove_pose_clip': '{id: 片段id}',
}


def contact_ids(document, scope=None, *, offset=0., selected_ids=(), fallback_range=None):
    if document is None:
        raise ValueError('请先打开工程或生成指法初稿')
    scope = {} if scope is None else scope
    if not isinstance(scope, dict) or set(scope) - {'ids', 'all', 'kind', 'time_range', 'hand', 'finger'}:
        raise ValueError('动作选择范围无效')
    if 'all' in scope and type(scope['all']) is not bool:
        raise ValueError('scope.all 必须是布尔值')
    if 'ids' in scope:
        ids = scope['ids']
        if not isinstance(ids, list) or any(type(i) is not int or not 1 <= i <= len(document) for i in ids):
            raise ValueError('动作编号无效，编号从 1 开始')
        selected = {i - 1 for i in ids}
    else:
        selected = set(range(len(document))) if scope.get('all') or 'time_range' in scope else set(selected_ids)
        if not selected and not scope.get('all') and 'time_range' not in scope:
            if fallback_range is None:
                raise ValueError('请明确动作编号、全部或时间范围')
            scope = dict(scope, time_range=fallback_range)
            selected = set(range(len(document)))
    for field, allowed in (('hand', ('left', 'right')), ('finger', TARGETS[1:])):
        if field in scope:
            if scope[field] not in allowed:
                raise ValueError('范围中的手或手指无效')
            selected = {i for i in selected if document.contact(i)[field] == scope[field]}
    if 'kind' in scope:
        if scope['kind'] not in ('tap', 'hold', 'drag', 'flick'):
            raise ValueError('未知动作类型')
        selected = {i for i in selected if document.contact(i)['kind'] == scope['kind']}
    if 'time_range' in scope:
        bounds = scope['time_range']
        if not isinstance(bounds, list) or len(bounds) != 2 or not all(map(finite, bounds)) or bounds[1] <= bounds[0]:
            raise ValueError('时间范围无效')
        low, high = [t - offset for t in bounds]
        selected = {i for i in selected if document.contact(i)['start'] < high and document.contact(i)['end'] > low}
    return document.sorted_ids(selected)


def apply_operations(document, operations, *, chart=None, offset=0., selected_ids=(), fallback_range=None):
    if not isinstance(operations, list) or len(operations) > 200:
        raise ValueError('每个候选需要一个至多 200 项的 operations 数组')
    doc = PlanDocument.from_dict(copy.deepcopy(document.snapshot()))
    for operation in operations:
        if (not isinstance(operation, dict) or not {'op', 'values'} <= operation.keys()
                or set(operation) - {'op', 'scope', 'values'}):
            raise ValueError('不支持的编辑操作')
        name, options = operation['op'], copy.deepcopy(operation['values'])
        if not isinstance(options, dict) or name not in OPERATION_PARAMETERS:
            raise ValueError('编辑操作或参数无效')
        if name in ('pose_defaults', 'pose_clip', 'remove_pose_clip'):
            pose = copy.deepcopy(doc.pose)
            if name == 'pose_defaults':
                if set(options) != {'hand', 'target', 'values'} or options['hand'] not in ('left', 'right') or options['target'] not in TARGETS:
                    raise ValueError('姿态目标无效')
                validate_values(options['values'], options['target'])
                key = options['hand'] + ':' + options['target']
                pose['defaults'][key] = pose['defaults'].get(key, {}) | options['values']
            elif name == 'pose_clip':
                options.setdefault('id', uuid.uuid4().hex)
                options['start'] -= offset
                options['end'] -= offset
                for key in options['keys']:
                    key['time'] -= offset
                index = next((i for i, c in enumerate(pose['clips']) if c['id'] == options['id']), None)
                if index is None:
                    pose['clips'].append(options)
                else:
                    pose['clips'][index] = options
            else:
                if set(options) != {'id'} or not any(c['id'] == options['id'] for c in pose['clips']):
                    raise ValueError('找不到姿态片段')
                pose['clips'] = [c for c in pose['clips'] if c['id'] != options['id']]
            doc.commit_pose('模型姿态修改', pose)
            continue
        ids = contact_ids(doc, operation.get('scope'), offset=offset, selected_ids=selected_ids, fallback_range=fallback_range)
        if not ids:
            raise ValueError('没有符合范围的动作；请先选择动作，或显式指定全部和筛选条件')
        if name == 'assign':
            doc.assign(ids, **options)
        elif name == 'alternate':
            if options.get('first', 'left') not in ('left', 'right'):
                raise ValueError('first 必须是 left 或 right')
            doc.alternate(ids, **options)
        elif name == 'swap':
            if options:
                raise ValueError('交换左右手不需要参数')
            doc.swap_hands(ids)
        elif name == 'lock':
            if any(type(v) is not bool for v in options.values()):
                raise ValueError('锁定参数必须是布尔值')
            doc.set_lock(ids, **options)
        elif name == 'paths':
            if 'only_hold' in options and type(options['only_hold']) is not bool:
                raise ValueError('only_hold 必须是布尔值')
            if 'rows' in options and (not isinstance(options['rows'], list) or any(type(i) is not int for i in options['rows'])):
                raise ValueError('控制点编号必须是整数数组')
            if any(key in options and options[key] is not None and not finite(options[key]) for key in ('x', 'y', 'tolerance')):
                raise ValueError('轨迹参数必须是有限数值')
            doc.edit_paths(ids, **options)
        elif name == 'release':
            if options.get('mode') == 'absolute':
                options['value'] -= offset
            doc.set_release(ids, **options)
        elif name == 'flick':
            if any(key in options and not finite(options[key]) for key in ('angle', 'length', 'duration_ms', 'lead_ms', 'release_ms')):
                raise ValueError('Flick 参数必须是有限数值')
            if 'note_id' in options and options['note_id'] is not None and type(options['note_id']) is not int:
                raise ValueError('音符编号必须是整数')
            if chart is None:
                raise ValueError('修改 Flick 需要关联原谱面')
            from flowphi.ui.gestures import flick_contacts, flick_defaults
            entries = flick_contacts(doc, chart, ids)
            if not entries:
                raise ValueError('所选动作没有 Flick')
            found = False
            for i, contact, flicks in entries:
                chosen = [f for f in flicks if options.get('note_id') is None or f[0] == options['note_id']]
                if not chosen:
                    continue
                defaults = flick_defaults(contact, chosen, chart)
                defaults['length'] /= 100
                doc.set_flicks([i], chart, **(defaults | options))
                found = True
            if not found:
                raise ValueError('所选动作中没有指定的 Flick 音符')
        elif name == 'touch':
            doc.set_touch_mode(ids, **options)
        elif name == 'contact':
            if set(options) - {'hand', 'finger', 'start', 'end', 'points', 'assignment_locked', 'path_locked', 'touch_mode'}:
                raise ValueError('未知动作参数')
            for field in ('start', 'end'):
                if field in options:
                    options[field] -= offset
            if 'points' in options:
                options['points'] = [[t - offset, x, y] for t, x, y in options['points']]
                options.setdefault('path_locked', True)
            for field in ('assignment_locked', 'path_locked'):
                if field in options and type(options[field]) is not bool:
                    raise ValueError('锁定参数必须是布尔值')
            doc.commit('模型动作修改', {i: doc.edits.get(i, {}) | options for i in ids})
    return doc

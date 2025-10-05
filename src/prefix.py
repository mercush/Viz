import pandas as pd

def process_dataset_url(dataset_url: str) -> dict:
    """Return a dict mapping dataset URLs to their processed data for VegaLite visualization.
    Args:
        dataset_url: URL of the dataset to process

    Returns:
        Dict with 'fields' key containing list of field dicts with 'name', 'type, 'distinctValues'
    """
    # Read the CSV file
    df = pd.read_csv(dataset_url)

    # Process each field
    fields = []
    for col in df.columns:
        field_info = {
            'name': col,
            'type': infer_vegalite_type(df[col]),
            'distinctValues': int(df[col].nunique())
        }
        fields.append(field_info)

    return {'fields': fields}

def infer_vegalite_type(series: pd.Series) -> str:
    """Infer VegaLite field type from pandas Series.

    Args:
        series: Pandas Series to infer type from

    Returns:
        One of 'quantitative', 'nominal', 'ordinal', 'temporal'
    """
    # Check if temporal
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'temporal'

    # Check if numeric
    if pd.api.types.is_numeric_dtype(series):
        return 'quantitative'

    # Default to nominal for strings/objects
    return 'nominal'
def complete_check(bytes_list):
    """
    Check if a byte sequence represents a complete JSON object.

    Args:
        bytes_list: List of bytes or string representing JSON

    Returns:
        bool: True if JSON is complete and nothing follows, False otherwise
    """
    # Convert bytes to string if needed
    if isinstance(bytes_list, (list, bytes)):
        spec = bytes(bytes_list).decode('utf-8') if isinstance(bytes_list, list) else bytes_list.decode('utf-8')
    else:
        spec = bytes_list

    try:
        state = parse_json_state(spec)
        if state is None:
            return False

        # Check if we're at the root level (all structures closed)
        if len(state['stackArray']) != 1:
            return False

        # Check if we're not in the middle of a string
        if state['inString']:
            return False

        # Check if the root has any content (not empty)
        root = state['stackArray'][0]
        if not root.get('keys') and state['lastChar'] is None:
            return False

        return True
    except Exception:
        return False


def prefix_check(bytes_list, dataset):
    """
    Check if a byte sequence is a valid prefix of a Vega-Lite specification.

    Args:
        bytes_list: List of bytes or string representing partial Vega-Lite spec
        dataset: Dict with 'fields' key containing list of field dicts with 'name', 'type', 'distinctValues'

    Returns:
        bool: True if valid prefix, False otherwise
    """
    # Convert bytes to string if needed
    if isinstance(bytes_list, (list, bytes)):
        spec = bytes(bytes_list).decode('utf-8') if isinstance(bytes_list, list) else bytes_list.decode('utf-8')
    else:
        spec = bytes_list

    try:
        state = parse_json_state(spec)
        if state is None:
            return False

        if not check_json_structure(state, spec):
            return False

        if not check_vegalite_semantics(state, dataset):
            return False

        return True
    except Exception:
        return False


def parse_json_state(spec):
    """Parse the partial JSON and return state information."""
    escape_next = False
    stack_array = [{'type': 'root', 'keys': [], 'expectingKey': False, 'expectingValue': False}]

    state = {
        'inString': False,
        'currentString': '',
        'stackArray': stack_array,
        'lastChar': None
    }

    for char in spec:
        if escape_next:
            escape_next = False
            if state['inString']:
                state['currentString'] += char
            continue
        
        if char == '\\' and state['inString']:
            escape_next = True
            state['currentString'] += char
            continue
        
        if char == '"':
            if state['inString']:
                top = stack_array[-1]
                if top.get('waitingForKey'):
                    top['lastKey'] = state['currentString']
                    top['waitingForKey'] = False
                    top['expectingColon'] = True
                elif top.get('expectingValue'):
                    if top.get('lastKey'):
                        top['keys'].append({'key': top['lastKey'], 'value': state['currentString']})
                    top['expectingValue'] = False
                    top['lastKey'] = None
                state['inString'] = False
                state['currentString'] = ''
            else:
                state['inString'] = True
                state['currentString'] = ''
                
                top = stack_array[-1]
                if top['type'] == 'object' and top.get('expectingKey'):
                    top['waitingForKey'] = True
                    top['expectingKey'] = False
            state['lastChar'] = '"'
            continue
        
        if state['inString']:
            state['currentString'] += char
            continue
        
        if char.isspace():
            continue
        
        if char == '{':
            stack_array.append({
                'type': 'object',
                'keys': [],
                'expectingKey': True,
                'waitingForKey': False,
                'expectingColon': False,
                'expectingValue': False,
                'lastKey': None
            })
            state['lastChar'] = '{'
        elif char == '}':
            if len(stack_array) > 1:
                stack_array.pop()
                # Update parent state after closing the object
                if len(stack_array) > 0:
                    parent = stack_array[-1]
                    if parent.get('expectingValue'):
                        parent['expectingValue'] = False
                        parent['lastKey'] = None
            state['lastChar'] = '}'
        elif char == '[':
            stack_array.append({
                'type': 'array',
                'items': [],
                'expectingValue': True
            })
            state['lastChar'] = '['
        elif char == ']':
            if len(stack_array) > 1:
                stack_array.pop()
                # Update parent state after closing the array
                if len(stack_array) > 0:
                    parent = stack_array[-1]
                    if parent.get('expectingValue'):
                        parent['expectingValue'] = False
                        parent['lastKey'] = None
            state['lastChar'] = ']'
        elif char == ':':
            top = stack_array[-1]
            if top.get('expectingColon'):
                top['expectingColon'] = False
                top['expectingValue'] = True
            state['lastChar'] = ':'
        elif char == ',':
            top = stack_array[-1]
            if top['type'] == 'object':
                top['expectingKey'] = True
                top['expectingValue'] = False
                top['expectingColon'] = False
                top['lastKey'] = None
            elif top['type'] == 'array':
                top['expectingValue'] = True
            state['lastChar'] = ','
        else:
            state['lastChar'] = char
    
    return state


def check_json_structure(state, spec):
    """Check if JSON structure is valid."""
    if state['inString']:
        return True
    
    top = state['stackArray'][-1]
    
    # Root level must start with { or [
    if len(state['stackArray']) == 1 and spec.strip():
        if not spec.strip().startswith('{') and not spec.strip().startswith('['):
            return False
    
    if top['type'] == 'object':
        if top.get('expectingColon') and state['lastChar'] != '"':
            return True
    
    return True


def check_vegalite_semantics(state, dataset):
    """Check Vega-Lite semantic constraints."""
    valid_mark_types = ['point', 'line', 'bar', 'area', 'rect', 'rule', 'text', 'tick', 'circle', 'square']
    valid_aggregates = ['mean', 'sum', 'count', 'min', 'max', 'median', 'stdev', 'variance']
    valid_channels = ['x', 'y', 'color', 'size', 'shape', 'opacity', 'column', 'row']
    valid_vega_types = ['quantitative', 'nominal', 'ordinal', 'temporal']
    channel_properties = ['field', 'type', 'aggregate', 'bin', 'timeUnit', 'scale', 'axis', 'legend', 'title']
    top_level_properties = ['mark', 'encoding', 'data', 'width', 'height', 'title', 'description', 'transform', 'config']
    
    if state['inString']:
        top = state['stackArray'][-1]
        
        if top.get('waitingForKey'):
            context = get_semantic_context(state['stackArray'], len(state['stackArray']) - 1, valid_channels)
            valid_keys = get_valid_keys(context, valid_channels, channel_properties, top_level_properties)
            partial = state['currentString']

            # Empty valid_keys means accept any key
            if valid_keys and partial and not any(k.startswith(partial) for k in valid_keys):
                return False
        
        elif top.get('expectingValue'):
            context = get_semantic_context(state['stackArray'], len(state['stackArray']) - 1, valid_channels)
            valid_values = get_valid_values(context, dataset, top.get('lastKey'), 
                                           valid_mark_types, valid_vega_types, valid_aggregates)
            partial = state['currentString']
            
            if valid_values and partial and not any(v.startswith(partial) for v in valid_values):
                return False
    else:
        top = state['stackArray'][-1]
        if top and (top.get('expectingValue') or top.get('expectingKey') or top.get('expectingColon')):
            return True
        
        # Check completed keys and values
        for i in range(1, len(state['stackArray'])):
            level = state['stackArray'][i]
            if level['type'] == 'object' and level.get('keys'):
                # Get context for this level, excluding lastKey from this level
                # (lastKey represents an incomplete key-value pair still being processed)
                context = get_semantic_context(state['stackArray'], i, valid_channels, exclude_last_key_at_level=i)
                valid_keys = get_valid_keys(context, valid_channels, channel_properties, top_level_properties)

                for entry in level['keys']:
                    # Empty valid_keys means accept any key
                    if valid_keys and entry['key'] not in valid_keys:
                        return False

                    valid_values = get_valid_values(context, dataset, entry['key'],
                                                   valid_mark_types, valid_vega_types, valid_aggregates)
                    if valid_values and entry['value'] not in valid_values:
                        return False
    
    return True


def get_semantic_context(stack_array, level, valid_channels, exclude_last_key_at_level=None):
    """Get semantic context at a given stack level."""
    path_keys = []
    current_channel = None
    inside_channel_object = False

    for i in range(level + 1):
        lvl = stack_array[i]

        if lvl['type'] == 'object':
            # Only include lastKey if we're not excluding it at this level
            if lvl.get('lastKey') and i != exclude_last_key_at_level:
                path_keys.append(lvl['lastKey'])
                if lvl['lastKey'] in valid_channels:
                    current_channel = lvl['lastKey']

            if lvl.get('keys'):
                for entry in lvl['keys']:
                    path_keys.append(entry['key'])
                    if entry['key'] in valid_channels:
                        current_channel = entry['key']
    
    if level >= 1:
        parent_level = stack_array[level - 1]
        if parent_level['type'] == 'object' and parent_level.get('lastKey') in valid_channels:
            inside_channel_object = True
            current_channel = parent_level['lastKey']
    
    return {
        'pathKeys': path_keys,
        'hasEncoding': 'encoding' in path_keys,
        'hasMark': 'mark' in path_keys,
        'currentChannel': current_channel,
        'insideChannelObject': inside_channel_object,
        'inFacetChannel': current_channel in ['column', 'row'] if current_channel else False
    }


def get_valid_keys(context, valid_channels, channel_properties, top_level_properties):
    """Get valid keys for current context."""
    if context['insideChannelObject']:
        return channel_properties
    elif context['hasEncoding'] and not context['insideChannelObject']:
        return valid_channels
    elif 'data' in context['pathKeys']:
        # Inside data object, allow any keys (url, values, format, etc.)
        return []  # Empty list means accept any key
    else:
        return top_level_properties


def get_valid_values(context, dataset, key, valid_mark_types, valid_vega_types, valid_aggregates):
    """Get valid values for a given key."""
    if not key:
        return []
    
    if key == 'mark':
        return valid_mark_types
    elif key == 'field':
        if context['inFacetChannel']:
            return [f['name'] for f in dataset['fields'] if f['distinctValues'] <= 50]
        return [f['name'] for f in dataset['fields']]
    elif key == 'type':
        return valid_vega_types
    elif key == 'aggregate':
        return valid_aggregates
    
    return []


# Example usage
if __name__ == "__main__":
    dataset = {
        'fields': [
            {'name': 'Horsepower', 'type': 'number', 'distinctValues': 94},
            {'name': 'Acceleration', 'type': 'number', 'distinctValues': 96},
            {'name': 'Year', 'type': 'number', 'distinctValues': 13},
            {'name': 'Origin', 'type': 'string', 'distinctValues': 3},
            {'name': 'Name', 'type': 'string', 'distinctValues': 303},
        ]
    }
    
    # Test cases
    test_cases = [
        ('{"mark": "p', True),  # Valid - partial "point"
        ('{"mark": "xyz', False),  # Invalid - no mark type starts with "xyz"
        ('{"encoding": {"x": {"field": "H', True),  # Valid - partial "Horsepower"
        ('{"encoding": {"x": {"field": "h', False),  # Invalid - case-sensitive
        ('{"mark": "point", "encoding"     :      {    "x": {"field":', True),  # Valid - expecting value
        ('{"encoding": {"column": {"field": "Name', False),  # Invalid - Name has too many values for faceting
    ]
    
    for spec, expected in test_cases:
        result = prefix_check(spec, dataset)
        status = "✓" if result == expected else "✗"
        print(f"{status} prefix_check({spec!r}) = {result} (expected {expected})")
    test_program = b"""{
  "data": {"url": "tests/dataset.csv"},
  "mark": "point",
  "encoding": {
    "x": {
      "field": "age"
    },
    "y": {
      "field": "height"
    }
  }
}
"""
    test_dataset = process_dataset_url("tests/dataset.csv")

    for i in range(len(test_program)):
        prefix = test_program[:i]
        is_complete = complete_check(prefix)
        is_valid_prefix = prefix_check(prefix, test_dataset)
        print(f"Byte {i}: prefix={is_valid_prefix}, complete={is_complete}")

def remove_sql_functions(sql_query):
    def remove_function(match):
        content = match.group(1)

        open_count = 0
        final_content = []
        current_param = []

        for char in content:
            if char == '(':
                open_count += 1
                if open_count > 0:
                    current_param.append(char)
            elif char == ')':
                open_count -= 1
                if open_count >= 0:
                    current_param.append(char)
            elif char == ',' and open_count == 0:
                final_content.append(''.join(current_param).strip())
                current_param = []
            else:
                current_param.append(char)

        if current_param:
            final_content.append(''.join(current_param).strip())

        return final_content[0] if final_content else ''

    convert_pattern = r'CONVERT_TZ\((.*?)\)'
    sql_query = re.sub(convert_pattern, lambda m: remove_function(m), sql_query, flags=re.IGNORECASE)

    coalesce_pattern = r'COALESCE\((.*?)\)'
    sql_query = re.sub(coalesce_pattern, lambda m: remove_function(m), sql_query, flags=re.IGNORECASE)

    return sql_query
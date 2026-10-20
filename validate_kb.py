#!/usr/bin/env python3
"""Validate Wajhah JSON, the bundled schema, cross-links and optional original DOCX.

Uses only Python's standard library. The schema validator implements EVERY
keyword used by the bundled schema, and fails if an unsupported keyword appears.
It is deliberately not presented as a general-purpose JSON Schema engine.
"""
from pathlib import Path
from collections import Counter
from datetime import date
from urllib.parse import urlsplit
import argparse, hashlib, json, re, sys, zipfile, xml.etree.ElementTree as ET

COLLECTIONS=['sectors','source_documents','source_sections','authorities','services',
 'complaint_scenarios','taxonomies','taxonomy_values','channels','routing_rules',
 'forms','form_fields','form_controls','form_rules','warnings','gaps','research_metadata']
SLUGS=['media','health','investment','education','hrsd']
ALLOWED_KEYWORDS={'$schema','$id','$defs','$ref','title','description','type','properties','required',
 'additionalProperties','items','minItems','maxItems','uniqueItems','minLength','maxLength',
 'minimum','maximum','pattern','enum','const','anyOf','format'}

class ValidationError(Exception): pass
def require(test,message):
    if not test:raise ValidationError(message)
def no_dupes(pairs):
    out={}
    for k,v in pairs:
        if k in out:raise ValidationError(f'Duplicate JSON property: {k}')
        out[k]=v
    return out
def invalid_constant(s):raise ValidationError(f'Invalid JSON numeric constant: {s}')
def load(path):
    with path.open('r',encoding='utf-8') as f:
        return json.load(f,object_pairs_hook=no_dupes,parse_constant=invalid_constant)
def is_type(x,t):
    return {'null':x is None,'boolean':isinstance(x,bool),'string':isinstance(x,str),
      'number':isinstance(x,(int,float)) and not isinstance(x,bool),
      'integer':isinstance(x,int) and not isinstance(x,bool),
      'array':isinstance(x,list),'object':isinstance(x,dict)}[t]
def check_supported(s,path='$'):
    require(isinstance(s,dict),f'Invalid schema object at {path}')
    require(set(s)<=ALLOWED_KEYWORDS,f'Unsupported schema keywords at {path}: {set(s)-ALLOWED_KEYWORDS}')
    for k in ['properties','$defs']:
        for key,v in s.get(k,{}).items():check_supported(v,f'{path}/{k}/{key}')
    for k in ['items','additionalProperties']:
        if isinstance(s.get(k),dict):check_supported(s[k],f'{path}/{k}')
    for i,v in enumerate(s.get('anyOf',[])):check_supported(v,f'{path}/anyOf/{i}')
    if 'required' in s and 'properties' in s:
        require(set(s['required'])<=set(s['properties']),f'Schema required property not defined at {path}')
    if '$ref' in s:require(s['$ref'].startswith('#/$defs/'),f'External schema reference unsupported: {s["$ref"]}')

def schema_validate(x,s,root,path='$'):
    if '$ref' in s:
        name=s['$ref'][len('#/$defs/'):]
        require(name in root['$defs'],f'Unknown schema definition: {name}')
        return schema_validate(x,root['$defs'][name],root,path)
    if 'anyOf' in s:
        errors=[]
        for sub in s['anyOf']:
            try:schema_validate(x,sub,root,path);break
            except ValidationError as e:errors.append(str(e))
        else:raise ValidationError(f'{path}: no anyOf branch matched: {errors}')
    if 'type' in s:
        types=s['type'] if isinstance(s['type'],list) else [s['type']]
        require(any(is_type(x,t) for t in types),f'{path}: expected {types}, got {type(x).__name__}')
    if 'const' in s:require(x==s['const'],f'{path}: incorrect constant {x!r}')
    if 'enum' in s:require(x in s['enum'],f'{path}: unknown enum {x!r}')
    if isinstance(x,dict):
        for key in s.get('required',[]):require(key in x,f'{path}: missing property {key}')
        props=s.get('properties',{})
        for k,v in x.items():
            if k in props:schema_validate(v,props[k],root,f'{path}/{k}')
            elif s.get('additionalProperties') is False:raise ValidationError(f'{path}: unrecognized property {k}')
            elif isinstance(s.get('additionalProperties'),dict):schema_validate(v,s['additionalProperties'],root,f'{path}/{k}')
    if isinstance(x,list):
        if 'minItems' in s:require(len(x)>=s['minItems'],f'{path}: too few items')
        if 'maxItems' in s:require(len(x)<=s['maxItems'],f'{path}: too many items')
        if s.get('uniqueItems'):require(len({json.dumps(v,sort_keys=True) for v in x})==len(x),f'{path}: duplicate items')
        if 'items' in s:
            for i,v in enumerate(x):schema_validate(v,s['items'],root,f'{path}/{i}')
    if isinstance(x,str):
        if 'minLength' in s:require(len(x)>=s['minLength'],f'{path}: string too short')
        if 'maxLength' in s:require(len(x)<=s['maxLength'],f'{path}: string too long')
        if 'pattern' in s:require(re.search(s['pattern'],x) is not None,f'{path}: pattern mismatch')
        if s.get('format')=='date':
            try:date.fromisoformat(x)
            except ValueError:raise ValidationError(f'{path}: invalid date')
            require(bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}',x)),f'{path}: noncanonical date')
        if s.get('format')=='uri':
            u=urlsplit(x);require(u.scheme in ['http','https'] and bool(u.netloc) and not re.search(r'\s',x),f'{path}: invalid HTTP(S) URL')
    if isinstance(x,(int,float)) and not isinstance(x,bool):
        if 'minimum' in s:require(x>=s['minimum'],f'{path}: below minimum')
        if 'maximum' in s:require(x<=s['maximum'],f'{path}: above maximum')

def walk(x,path='$'):
    if isinstance(x,dict):
        yield x,path
        for k,v in x.items():yield from walk(v,f'{path}/{k}')
    elif isinstance(x,list):
        for i,v in enumerate(x):yield from walk(v,f'{path}/{i}')

def semantic(kb):
    idx={};by={}
    for key in COLLECTIONS:
        by[key]={}
        for r in kb[key]:
            require(r['id'] not in idx,f'Duplicate ID: {r["id"]}')
            idx[r['id']]=r;by[key][r['id']]=r
    link_types={'source_document_id':'source_documents','source_document_ids':'source_documents',
      'source_section_ids':'source_sections','authority_id':'authorities','authority_ids':'authorities',
      'sector_id':'sectors','service_id':'services','scenario_ids':'complaint_scenarios',
      'supported_scenario_ids':'complaint_scenarios','taxonomy_id':'taxonomies','taxonomy_ids':'taxonomies',
      'value_ids':'taxonomy_values','taxonomy_value_ids':'taxonomy_values','parent_value_id':'taxonomy_values',
      'option_value_ids':'taxonomy_values','target_option_value_ids':'taxonomy_values',
      'form_id':'forms','form_ids':'forms','field_id':'form_fields','field_ids':'form_fields',
      'trigger_field_id':'form_fields','target_field_ids':'form_fields','hidden_field_ids':'form_fields',
      'dependency_field_ids':'form_fields','control_ids':'form_controls','rule_ids':'form_rules',
      'gap_ids':'gaps','channel_id':'channels','platform_channel_id':'channels','entry_channel_id':'channels',
      'official_website_channel_id':'channels','observed_website_channel_ids':'channels',
      'escalation_authority_id':'authorities','service_related_complaint_service_id':'services',
      'integrity_or_violation_report_service_id':'services'}
    refs=0
    for node,path in walk(kb):
        # Source payloads are literal historical claims, not normalized cross-links.
        if '/reported_claims' in path or '/source_claims' in path or '/research_log_reported' in path:continue
        for key,col in link_types.items():
            v=node.get(key)
            if v is None:continue
            values=v if isinstance(v,list) else [v]
            for ident in values:
                require(ident in by[col],f'{path}/{key}: missing {col} reference {ident}');refs+=1
        for ident in node.get('affected_record_ids',[]):require(ident in idx,f'{path}: gap targets missing ID {ident}')
        if 'provenance' in node:
            p=node['provenance']
            for s in p['source_section_ids']:
                require(by['source_sections'][s]['source_document_id'] in p['source_document_ids'],f'{path}: provenance file/section mismatch')
            for u in p['observed_website_channel_ids']:
                require(set(by['channels'][u]['provenance']['source_document_ids'])<=set(p['source_document_ids']),f'{path}: provenance URL source mismatch')
    for section in kb['source_sections']:
        l=section['locator'];doc=by['source_documents'][section['source_document_id']]
        require(l['paragraph_start']<=l['paragraph_end']<=doc['paragraph_count'],f'Invalid locator {section["id"]}')
    for s in kb['services']:
        require(s['entry_channel_id'] is not None,f'Service missing documented entry link: {s["id"]}')
        ch=by['channels'][s['entry_channel_id']]
        require(ch['url'] is not None and ch['authority_id']==s['authority_id'],f'Service authority/link mismatch: {s["id"]}')
        for fid in s['form_ids']:require(by['forms'][fid]['service_id']==s['id'],f'Service/form mismatch {fid}')
        for sc in s['supported_scenario_ids']:require(by['complaint_scenarios'][sc]['service_id']==s['id'],f'Service/scenario mismatch {sc}')
        require(s['eligibility']['criteria'] is None,f'Unexpected unsupported eligibility rule: {s["id"]}')
        require(s['follow_up_question_candidates'] is None,f'Invented source follow-up questions: {s["id"]}')
        require(s['agent_readiness']=='UNVERIFIED_END_TO_END',f'Unsupported agent readiness: {s["id"]}')
    for sc in kb['complaint_scenarios']:
        svc=by['services'][sc['service_id']]
        require(sc['authority_id']==svc['authority_id'],f'Scenario authority mismatch: {sc["id"]}')
        require(sc['channel_id'] is not None,f'Scenario missing source link: {sc["id"]}')
        require(sc['id'] in svc['supported_scenario_ids'],f'Orphan scenario: {sc["id"]}')
    for t in kb['taxonomies']:
        actual=[v['id'] for v in kb['taxonomy_values'] if v['taxonomy_id']==t['id']]
        require(actual==t['value_ids'],f'Taxonomy/value inventory mismatch: {t["id"]}')
        require(t['id'] in by['services'][t['service_id']]['taxonomy_ids'],f'Orphan taxonomy: {t["id"]}')
    for v in kb['taxonomy_values']:
        parent=v['parent_value_id']
        if parent:
            require(by['taxonomies'][by['taxonomy_values'][parent]['taxonomy_id']]['service_id']==by['taxonomies'][v['taxonomy_id']]['service_id'],f'Taxonomy parent service mismatch: {v["id"]}')
    for form in kb['forms']:
        for collection,key in [('form_fields','field_ids'),('form_controls','control_ids'),('form_rules','rule_ids')]:
            actual=[r['id'] for r in kb[collection] if r['form_id']==form['id']]
            require(actual==form[key],f'Form inventory mismatch: {form["id"]}/{key}')
        require(form['submission']['submitted_case_outcome'] is None,f'Invented submitted-case outcome: {form["id"]}')
        require(form['submission']['extra_otp_required'] is None,f'Unverified post-submit OTP policy: {form["id"]}')
        require(form['submission']['review_screen_exists'] is (True if form['id']=='form.education.complaint' else None),f'Review observation was overgeneralized: {form["id"]}')
    for field in kb['form_fields']:
        form=by['forms'][field['form_id']]
        for tv in field['option_value_ids'] or []:
            require(by['taxonomies'][by['taxonomy_values'][tv]['taxonomy_id']]['service_id']==form['service_id'],f'Field option service mismatch: {field["id"]}')
        for dep in field['dependency_field_ids'] or []:
            require(by['form_fields'][dep]['form_id']==field['form_id'],f'Cross-form dependency: {field["id"]}')
    for rule in kb['form_rules']:
        for fid in rule['target_field_ids']+([rule['trigger_field_id']] if rule['trigger_field_id'] else []):
            require(by['form_fields'][fid]['form_id']==rule['form_id'],f'Rule field/form mismatch: {rule["id"]}')
    for channel in kb['channels']:
        require((channel['url'] is None)!=(channel['contact_value'] is None),f'Channel must contain one URL/contact: {channel["id"]}')
        if channel['kind']=='DISCOVERY':require(channel['official_status']=='NON_OFFICIAL_DISCOVERY',f'Discovery marked official: {channel["id"]}')
    for col,key in [('taxonomy_values','parent_value_id'),('form_fields','dependency_field_ids')]:
        def visit(ident,active,done):
            require(ident not in active,f'Cyclic {key}: {ident}')
            if ident in done:return
            active.add(ident);v=by[col][ident][key];deps=v if isinstance(v,list) else [v] if v else []
            for dep in deps:visit(dep,active,done)
            active.remove(ident);done.add(ident)
        done=set()
        for ident in by[col]:visit(ident,set(),done)
    if 'auth.gmedia' in idx:
        require(idx['auth.gmedia']['name_ar']=='الهيئة العامة لتنظيم الإعلام' and idx['auth.gmedia']['kind']=='REGULATOR','Media regulator identity lost')
        require(not any(a['name_ar']=='وزارة الإعلام' for a in kb['authorities']),'Ministry of Media incorrectly used as recipient')
    if 'tax.education.main_service' in idx:require(len(idx['tax.education.main_service']['value_ids'])==34,'Education category count != 34')
    for fid in ['form.media.report','form.health.ticket','form.education.complaint','form.hrsd.call_center']:
        if fid in idx:require(idx[fid]['authentication']['nafath_required'] is None,f'Unverified Nafath requirement {fid}')
    require(kb['verified_at'] is None and not kb['verification']['website_verification_performed_by_converter'],'Unexpected live-verification claim')
    return {'records':len(idx),'references_checked':refs,'cycle_checks':'PASSED','known_unknown_checks':'PASSED'}

def source_text(path):
    ns='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    with zipfile.ZipFile(path) as z:
        require(z.testzip() is None,f'Corrupt DOCX: {path.name}')
        root=ET.fromstring(z.read('word/document.xml'));paras=[]
        for i,p in enumerate(root.findall('.//'+ns+'p'),1):
            txt=''.join(n.text or '' if n.tag in [ns+'t',ns+'delText',ns+'instrText']
                        else '\t' if n.tag==ns+'tab' else '\n' if n.tag==ns+'br' else '' for n in p.iter())
            paras.append((i,txt))
        return paras
def url_inventory(text):
    # Removes surrounding prose punctuation, never edits URL path/case/query/fragment.
    return sorted({u.rstrip(',):') for u in re.findall(r'https?://[^\s"<>]+',text)})
def parse_first_array(text):
    d=json.JSONDecoder();p=text.find('[')
    require(p>=0,'Source array missing')
    return d.raw_decode(text[p:])[0]
def source_audit(kb,source_dir):
    checks=[]
    for doc in kb['source_documents']:
        path=source_dir/doc['filename'];require(path.is_file(),f'Missing original: {path}')
        require(hashlib.sha256(path.read_bytes()).hexdigest()==doc['sha256'],f'Source hash differs: {doc["filename"]}')
        ps=source_text(path);require(len(ps)==doc['paragraph_count'],f'Paragraph count differs: {path.name}')
        sections=[s for s in kb['source_sections'] if s['source_document_id']==doc['id']]
        for s in sections:
            a,z=s['locator']['paragraph_start'],s['locator']['paragraph_end']
            raw='\n'.join(txt for i,txt in ps if a<=i<=z and txt.strip())
            require(raw==s['raw_text'],f'Source excerpt differs: {s["id"]}')
        expected_urls=url_inventory('\n'.join(t for _,t in ps))
        actual_urls=sorted(ch['url'] for ch in kb['channels'] if ch['url'] and doc['id'] in ch['provenance']['source_document_ids'])
        require(expected_urls==actual_urls,f'URL inventory mismatch {path.name}: expected={expected_urls}, actual={actual_urls}')
        fields=next((s for s in sections if s['section_name']=='fields'),None)
        labels=[];options=[]
        out_fields=[f for f in kb['form_fields'] if doc['id'] in f['provenance']['source_document_ids']]
        out_controls=[c for c in kb['form_controls'] if doc['id'] in c['provenance']['source_document_ids']]
        value_index={x['id']:x for x in kb['taxonomy_values']}
        if fields and doc['id'] in ['src.health','src.education','src.hrsd']:
            parsed=parse_first_array(fields['raw_text'])
            for fd in parsed:
                labels.append(fd['field_name_ar']);options.extend(fd.get('options',[]))
                actual=next((f for f in out_fields if f['label']==fd['field_name_ar']),None)
                require(actual is not None,f'Lost source field {fd["field_name_ar"]}')
                if fd.get('options'):
                    actual_values=[value_index[i]['source_labels'] for i in actual['option_value_ids'] or []]
                    require(len(actual_values)==len(fd['options']),f'Field option count mismatch {fd["field_name_ar"]}')
                    for expected,variants in zip(fd['options'],actual_values):
                        require(expected in variants,f'Field option linkage/order mismatch {fd["field_name_ar"]}: {expected}')
            for label in labels:require(any(f['label']==label for f in out_fields),f'Lost source field: {path.name}: {label}')
            out_labels={lab for v in kb['taxonomy_values'] if doc['id'] in v['provenance']['source_document_ids'] for lab in v['source_labels']}
            for option in options:require(option in out_labels,f'Lost source option: {path.name}: {option}')
        if doc['id']=='src.media':
            section=next(s for s in sections if s['section_name']=='form_steps')
            labels=parse_first_array(section['raw_text'])[0]['fields']
            for label in labels:require(any(f['label']==label for f in out_fields),f'Lost Media field: {label}')
            section=next(s for s in sections if s['section_name']=='taxonomy')
            out_labels={lab for v in kb['taxonomy_values'] if doc['id'] in v['provenance']['source_document_ids'] for lab in v['source_labels']}
            for row in parse_first_array(section['raw_text']):
                for key in ['level_1','level_2']:
                    if row[key] not in ['UNKNOWN','']:require(row[key] in out_labels,f'Lost Media category: {row[key]}')
        if doc['id']=='src.investment':
            fieldlabels={f['label'] for f in out_fields};controllabels={c['label'] for c in out_controls}
            for n,txt in ps:
                if 33<=n<=46 and txt.strip():
                    label=txt[2:]
                    label={'Country code selector (+966 observed)':'Country code selector',
                      'Reason For Contact (Complaint / Inquiry)':'Reason For Contact',
                      'Submit button':'Submit'}.get(label,label)
                    require(label in fieldlabels or label in controllabels,f'Lost investment field/control: {label}')
                    labels.append(label)
                if 51<=n<=56:
                    options.append(txt.strip('- ').strip())
            sub=next(f for f in out_fields if f['label']=='Select Sub-Category')
            actual=[value_index[i]['label'] for i in sub['option_value_ids'] or []]
            require(actual==options,'Investment technical subcategory linkage/order mismatch')
            raw=next(txt for n,txt in ps if n==78).split('include: ',1)[1].rstrip('.')
            for label in [x.strip() for x in raw.split(',')]:
                if label=='two consentdeclarations':
                    require(len([f for f in out_fields if f['form_id']=='form.investment.integrity' and f['field_kind']=='CONSENT'])==2,'Lost integrity consent declarations')
                else:require(label in fieldlabels or label in controllabels,f'Lost integrity field/control: {label}')
                labels.append(label)
        if doc['id']!='src.investment':
            section=next(s for s in sections if s['section_name']=='scenarios')
            original_scenarios=parse_first_array(section['raw_text'])
            out_scenarios=[s for s in kb['complaint_scenarios'] if doc['id'] in s['provenance']['source_document_ids']]
            require(len(original_scenarios)==len(out_scenarios),f'Lost/invented scenarios: {path.name}')
            channel_index={c['id']:c for c in kb['channels']}
            for original in original_scenarios:
                actual=next((s for s in out_scenarios if s['original_scenario_id']==original['scenario_id']),None)
                require(actual is not None,f'Lost scenario: {original["scenario_id"]}')
                for key in ['issue_domain','description','example_user_message_ar']:
                    require(actual[key]==original[key],f'Scenario source text differs: {original["scenario_id"]}/{key}')
                require(channel_index[actual['channel_id']]['url']==original['direct_link'],f'Scenario source URL differs: {original["scenario_id"]}')
        checks.append({'source_document_id':doc['id'],'sha256':'MATCHED','source_excerpts':'EXACTLY_MATCHED',
          'paragraphs_read':len(ps),'unique_urls_checked':len(expected_urls),'source_fields_checked':len(labels),
          'source_field_option_occurrences_checked':len(options)})
    return checks

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',nargs='?',default=str(Path(__file__).resolve().parent))
    parser.add_argument('--source-dir',type=Path,help='Folder containing the five original DOCX files; performs source hashes/excerpts/URL/field/option checks')
    parser.add_argument('--write-report',type=Path,help='Save machine-readable validation results')
    args=parser.parse_args();folder=Path(args.directory)
    schema=load(folder/'wajhah_kb.schema.json');check_supported(schema)
    files=[folder/f'wajhah_{s}.json' for s in SLUGS]+[folder/'wajhah_knowledge_base.json']
    datasets=[];results=[]
    for path in files:
        kb=load(path);schema_validate(kb,schema,schema);sem=semantic(kb)
        results.append({'file':path.name,'strict_json_load':'PASSED','json_schema':'PASSED',
                        'semantic_consistency':'PASSED',**sem});datasets.append(kb)
    combined=datasets[-1]
    require(len(combined['source_documents'])==5,'Combined dataset must represent exactly five inputs')
    for col in COLLECTIONS:
        require(combined[col]==[r for kb in datasets[:-1] for r in kb[col]],f'Combined/separate mismatch in {col}')
    require(len(combined['complaint_scenarios'])==13,'Expected 13 source-authored scenarios')
    checked_sources=source_audit(combined,args.source_dir) if args.source_dir else None
    extra_json=[]
    for name in ['conversion_audit.json','validation_report.json']:
        p=folder/name
        if p.is_file():load(p);extra_json.append(name)
    report={'status':'PASSED','schema_engine':'STDLIB_IMPLEMENTATION_OF_ALL_BUNDLED_SCHEMA_KEYWORDS',
      'files':results,'combined_equals_separate_union':True,'original_source_audit':checked_sources,
      'additional_json_loaded':extra_json,'counts':{c:len(combined[c]) for c in COLLECTIONS},
      'limits':'These checks establish JSON/schema/reference/source-extraction consistency, not official correctness, current website availability, chatbot accuracy or successful submission.'}
    if args.write_report:args.write_report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'files_validated':len(results),'counts':report['counts'],
      'original_source_audit_performed':checked_sources is not None},ensure_ascii=False))

if __name__=='__main__':
    try:main()
    except (ValidationError,ValueError,KeyError,OSError,json.JSONDecodeError) as e:
        print(f'VALIDATION FAILED: {e}',file=sys.stderr);sys.exit(1)

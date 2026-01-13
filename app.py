"""
dashboard_mantenimientos_con_calendario.py
DASHBOARD
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import warnings
import calendar
from dateutil.relativedelta import relativedelta
warnings.filterwarnings('ignore')

# ============================================
# 1. URLs (REEMPLAZA CON TUS URLS REALES)
# ============================================

URL_MANTENIMIENTOS = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSI2Ujvaj6GKzQDfhFXZbZ3bzRRtUNluxlPLVQuyruijv4ILq6jMWasYR44BRr4lLxlUg9ZBU28FUek/pub?gid=424956520&single=true&output=csv"
URL_HISTORICO = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTIRp3QEfyTn35RIDPHN0sS9mQYCNeQCrJdhZY_DGLkkUfcwaXPCvIlkQvWJ3OzDrKNLmUZQ9HeL83h/pub?gid=216802721&single=true&output=csv"
URL_CATALOGO = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSI2Ujvaj6GKzQDfhFXZbZ3bzRRtUNluxlPLVQuyruijv4ILq6jMWasYR44BRr4lLxlUg9ZBU28FUek/pub?gid=41327256&single=true&output=csv"

# ============================================
# 2. FUNCIONES AUXILIARES MEJORADAS
# ============================================

def obtener_ultimo_mantenimiento(df_mto, unidad, tipo_servicio=None):
    """Obtiene el último mantenimiento de una unidad (opcionalmente filtrado por tipo)"""
    df_unidad = df_mto[df_mto['Nombre Unidad'] == unidad].copy()
    
    if tipo_servicio:
        mask = df_unidad['TIPO DE SERVICIO'].str.contains(tipo_servicio, case=False, na=False)
        df_unidad = df_unidad[mask]
    
    if len(df_unidad) == 0:
        return None, None, None
    
    # Ordenar por fecha descendente y tomar el último
    df_unidad = df_unidad.sort_values('FECHA', ascending=False)
    ultimo = df_unidad.iloc[0]
    
    return ultimo['FECHA'], ultimo['KILOMETRAJE'], ultimo['TIPO DE SERVICIO']

def obtener_ultimos_mantenimientos_por_tipo(df_mto, unidad):
    """Obtiene los últimos mantenimientos por tipo de servicio"""
    if 'TIPO DE SERVICIO' not in df_mto.columns:
        return {'general': obtener_ultimo_mantenimiento(df_mto, unidad)}
    
    tipos_unicos = df_mto['TIPO DE SERVICIO'].dropna().unique()
    resultados = {}
    
    for tipo in tipos_unicos:
        fecha, km, servicio = obtener_ultimo_mantenimiento(df_mto, unidad, tipo)
        if fecha and km:
            resultados[str(tipo).lower()] = {
                'fecha': fecha,
                'kilometraje': km,
                'servicio': servicio
            }
    
    return resultados

def obtener_ultimo_kilometraje(df_hist, unidad):
    """Obtiene el último kilometraje registrado en el histórico"""
    df_unidad = df_hist[df_hist['Nombre Unidad'] == unidad].copy()
    
    if len(df_unidad) == 0:
        return None, None
    
    df_unidad = df_unidad.sort_values('Fecha', ascending=False)
    ultimo = df_unidad.iloc[0]
    
    return ultimo['Kilometraje KM'], ultimo['Fecha']

def calcular_km_promedio_diario_mejorado(df_hist, unidad, dias_max=90):
    """Calcula KM promedio diario considerando solo los últimos N días"""
    df_unidad = df_hist[df_hist['Nombre Unidad'] == unidad].copy()
    
    if len(df_unidad) < 2:
        return 100
    
    # Ordenar por fecha descendente y tomar últimos registros
    df_unidad = df_unidad.sort_values('Fecha', ascending=False)
    
    # Tomar solo los últimos N días de registros
    fecha_reciente = df_unidad.iloc[0]['Fecha']
    fecha_limite = fecha_reciente - timedelta(days=dias_max)
    df_reciente = df_unidad[df_unidad['Fecha'] >= fecha_limite]
    
    if len(df_reciente) < 2:
        # Si no hay suficientes datos recientes, usar todos
        df_reciente = df_unidad
    
    # Agrupar por día (último registro del día)
    df_reciente['Fecha_Dia'] = df_reciente['Fecha'].dt.date
    df_por_dia = df_reciente.sort_values('Fecha', ascending=False)
    df_por_dia = df_por_dia.drop_duplicates(subset=['Fecha_Dia'], keep='first')
    df_por_dia = df_por_dia.sort_values('Fecha_Dia')
    
    if len(df_por_dia) < 2:
        return 100
    
    # Calcular diferencias
    total_km = 0
    total_dias = 0
    
    for i in range(1, len(df_por_dia)):
        fecha_actual = df_por_dia.iloc[i]['Fecha_Dia']
        fecha_anterior = df_por_dia.iloc[i-1]['Fecha_Dia']
        km_actual = df_por_dia.iloc[i]['Kilometraje KM']
        km_anterior = df_por_dia.iloc[i-1]['Kilometraje KM']
        
        dias_dif = (fecha_actual - fecha_anterior).days
        km_dif = km_actual - km_anterior
        
        if dias_dif > 0 and km_dif > 0:
            total_km += km_dif
            total_dias += dias_dif
    
    if total_dias > 0 and total_km > 0:
        return total_km / total_dias
    else:
        return 100

# ============================================
# 3. FUNCIONES PARA CALENDARIO
# ============================================

def crear_vista_calendario(df_final, df_tipo_unidad=None):
    """Crea vista de calendario por mes con Tipo Unidad"""
    
    # Filtrar unidades con fecha estimada
    df_con_fecha = df_final[df_final['Fecha Estimada Mantenimiento'] != ''].copy()
    
    if len(df_con_fecha) == 0:
        return pd.DataFrame()
    
    # Convertir fechas a datetime
    df_con_fecha['Fecha_Estimada_DT'] = pd.to_datetime(
        df_con_fecha['Fecha Estimada Mantenimiento'], 
        dayfirst=True,
        errors='coerce'
    )
    
    # Extraer mes y año
    df_con_fecha['Mes'] = df_con_fecha['Fecha_Estimada_DT'].dt.month
    df_con_fecha['Año'] = df_con_fecha['Fecha_Estimada_DT'].dt.year
    df_con_fecha['Mes_Año'] = df_con_fecha['Fecha_Estimada_DT'].dt.strftime('%Y-%m')
    df_con_fecha['Mes_Nombre'] = df_con_fecha['Fecha_Estimada_DT'].dt.strftime('%B')
    df_con_fecha['Dia'] = df_con_fecha['Fecha_Estimada_DT'].dt.day
    df_con_fecha['Semana'] = df_con_fecha['Fecha_Estimada_DT'].dt.isocalendar().week
    
    # Agregar Tipo Unidad si se proporciona
    if df_tipo_unidad is not None and not df_tipo_unidad.empty:
        df_con_fecha = pd.merge(
            df_con_fecha,
            df_tipo_unidad[['Unidad', 'Tipo Unidad']],
            on='Unidad',
            how='left'
        )
        df_con_fecha['Tipo Unidad'] = df_con_fecha['Tipo Unidad'].fillna('No especificado')
    
    return df_con_fecha

def crear_vista_por_mes(df_con_fecha, meses_a_mostrar=6):
    """Crea vista agrupada por mes"""
    
    if len(df_con_fecha) == 0:
        return pd.DataFrame()
    
    hoy = datetime.now()
    meses = []
    
    for i in range(meses_a_mostrar):
        fecha = hoy + relativedelta(months=i)
        meses.append(fecha.strftime('%Y-%m'))
    
    df_futuro = df_con_fecha[df_con_fecha['Mes_Año'].isin(meses)].copy()
    
    if len(df_futuro) == 0:
        return pd.DataFrame()
    
    vista_mes = df_futuro.groupby(['Año', 'Mes', 'Mes_Nombre', 'Mes_Año']).agg({
        'Unidad': 'count',
        'Alerta': lambda x: list(x),
        'Estado': lambda x: list(x)
    }).reset_index()
    
    vista_mes = vista_mes.rename(columns={'Unidad': 'Cantidad_Unidades'})
    vista_mes = vista_mes.sort_values(['Año', 'Mes'])
    
    return vista_mes

def crear_vista_semanal(df_con_fecha, semanas_a_mostrar=8):
    """Crea vista por semana"""
    
    if len(df_con_fecha) == 0:
        return pd.DataFrame()
    
    hoy = datetime.now()
    semana_actual = hoy.isocalendar().week
    año_actual = hoy.year
    
    df_futuro = df_con_fecha[
        (df_con_fecha['Año'] == año_actual) & 
        (df_con_fecha['Semana'] >= semana_actual)
    ].copy()
    
    if len(df_futuro) == 0:
        return pd.DataFrame()
    
    semanas = sorted(df_futuro['Semana'].unique())[:semanas_a_mostrar]
    df_futuro = df_futuro[df_futuro['Semana'].isin(semanas)]
    
    vista_semana = df_futuro.groupby(['Año', 'Semana']).agg({
        'Unidad': 'count',
        'Alerta': lambda x: list(x),
        'Fecha_Estimada_DT': 'min'
    }).reset_index()
    
    vista_semana = vista_semana.rename(columns={'Unidad': 'Cantidad_Unidades'})
    
    def obtener_rango_semana(fila):
        year = fila['Año']
        week = fila['Semana']
        first_day = datetime.fromisocalendar(year, week, 1)
        last_day = first_day + timedelta(days=6)
        return f"{first_day.strftime('%d/%m')} - {last_day.strftime('%d/%m')}"
    
    vista_semana['Rango_Semana'] = vista_semana.apply(obtener_rango_semana, axis=1)
    vista_semana = vista_semana.sort_values(['Año', 'Semana'])
    
    return vista_semana

# ============================================
# 4. FUNCIÓN PRINCIPAL MEJORADA
# ============================================

@st.cache_data(ttl=300)
def cargar_y_calcular_mejorado():
    """Función principal mejorada que considera múltiples mantenimientos"""
    
    try:
        # Cargar datos
        df_mto = pd.read_csv(URL_MANTENIMIENTOS, encoding='utf-8')
        df_hist = pd.read_csv(URL_HISTORICO, encoding='utf-8')
        df_cat = pd.read_csv(URL_CATALOGO, encoding='utf-8')
        
        # Preparar datos
        df_mto['FECHA'] = pd.to_datetime(df_mto['FECHA'], dayfirst=True, errors='coerce')
        df_mto['KILOMETRAJE'] = pd.to_numeric(
            df_mto['KILOMETRAJE'].astype(str).str.replace(',', ''), errors='coerce'
        )
        
        df_hist['Fecha'] = pd.to_datetime(df_hist['Fecha'], dayfirst=True, errors='coerce')
        df_hist['Kilometraje KM'] = pd.to_numeric(
            df_hist['Kilometraje KM'].astype(str).str.replace(',', ''), errors='coerce'
        )
        
        df_cat['INTERVALO'] = pd.to_numeric(
            df_cat['INTERVALO'].astype(str).str.replace(',', ''), errors='coerce'
        )
        
        # EXTRAER TIPO UNIDAD del catálogo
        if 'TIPO UNIDAD' in df_cat.columns:
            df_tipo_unidad = df_cat[['Nombre Unidad', 'TIPO UNIDAD']].copy()
            df_tipo_unidad = df_tipo_unidad.rename(columns={
                'Nombre Unidad': 'Unidad',
                'TIPO UNIDAD': 'Tipo Unidad'
            })
            df_tipo_unidad['Tipo Unidad'] = df_tipo_unidad['Tipo Unidad'].fillna('').astype(str).str.strip()
            df_tipo_unidad['Tipo Unidad'] = df_tipo_unidad['Tipo Unidad'].replace('', 'No especificado')
        else:
            df_tipo_unidad = pd.DataFrame(columns=['Unidad', 'Tipo Unidad'])
        
        # Obtener todas las unidades únicas
        unidades = list(set(
            list(df_mto['Nombre Unidad'].unique()) + 
            list(df_hist['Nombre Unidad'].unique()) + 
            list(df_cat['Nombre Unidad'].unique())
        ))
        unidades = [u for u in unidades if pd.notna(u) and str(u).strip() != '']
        
        resultados = []
        
        for unidad in unidades:
            # ============================================
            # A. OBTENER ÚLTIMO MANTENIMIENTO PREVENTIVO
            # ============================================
            ultimo_preventivo = None
            ultimo_km_preventivo = None
            ultima_fecha_preventivo = None
            
            if 'TIPO DE SERVICIO' in df_mto.columns:
                # Filtrar solo preventivos
                mask = df_mto['TIPO DE SERVICIO'].str.contains('Preventivo', case=False, na=False)
                df_preventivos = df_mto[mask].copy()
                
                if not df_preventivos.empty:
                    df_unidad_preventivo = df_preventivos[df_preventivos['Nombre Unidad'] == unidad]
                    if not df_unidad_preventivo.empty:
                        df_unidad_preventivo = df_unidad_preventivo.sort_values('FECHA', ascending=False)
                        ultimo_preventivo = df_unidad_preventivo.iloc[0]
                        ultimo_km_preventivo = ultimo_preventivo['KILOMETRAJE']
                        ultima_fecha_preventivo = ultimo_preventivo['FECHA']
            else:
                # Si no hay columna de tipo, tomar el último mantenimiento en general
                df_unidad_mto = df_mto[df_mto['Nombre Unidad'] == unidad]
                if not df_unidad_mto.empty:
                    df_unidad_mto = df_unidad_mto.sort_values('FECHA', ascending=False)
                    ultimo_preventivo = df_unidad_mto.iloc[0]
                    ultimo_km_preventivo = ultimo_preventivo['KILOMETRAJE']
                    ultima_fecha_preventivo = ultimo_preventivo['FECHA']
            
            # ============================================
            # B. OBTENER ÚLTIMO KILOMETRAJE (HISTÓRICO)
            # ============================================
            km_actual, fecha_km_actual = obtener_ultimo_kilometraje(df_hist, unidad)
            
            # ============================================
            # C. OBTENER INTERVALO DEL CATÁLOGO
            # ============================================
            intervalo = None
            if not df_cat.empty:
                df_unidad_cat = df_cat[df_cat['Nombre Unidad'] == unidad]
                if not df_unidad_cat.empty:
                    intervalo = df_unidad_cat.iloc[0]['INTERVALO']
            
            # ============================================
            # D. CÁLCULOS (solo si tenemos datos suficientes)
            # ============================================
            if (ultimo_km_preventivo is not None and 
                pd.notna(ultimo_km_preventivo) and 
                km_actual is not None and 
                pd.notna(km_actual) and 
                intervalo is not None and 
                pd.notna(intervalo) and intervalo > 0):
                
                hoy = datetime.now()
                
                # Calcular KM promedio diario
                km_promedio_diario = calcular_km_promedio_diario_mejorado(df_hist, unidad)
                
                # Cálculos básicos
                dias_transcurridos = (hoy - ultima_fecha_preventivo).days if ultima_fecha_preventivo else 0
                km_recorridos = km_actual - ultimo_km_preventivo
                km_esperado = ultimo_km_preventivo + intervalo
                
                # % Avance
                if intervalo > 0:
                    porc_avance = (km_recorridos / intervalo) * 100
                else:
                    porc_avance = 0
                
                # Días Faltantes
                if km_promedio_diario > 0:
                    dias_faltantes = (intervalo - km_recorridos) / km_promedio_diario
                    dias_faltantes = max(0, dias_faltantes)
                else:
                    dias_faltantes = np.nan
                
                # Fecha Estimada
                fecha_estimada = None
                if pd.notna(dias_faltantes) and not np.isinf(dias_faltantes):
                    try:
                        fecha_estimada = hoy + timedelta(days=int(dias_faltantes))
                    except:
                        fecha_estimada = None
                
                # Alertas
                if porc_avance >= 100:
                    alerta = '🔴 ATRASADO'
                elif porc_avance >= 80:
                    alerta = '🟡 PROXIMO'
                else:
                    alerta = '🟢 EN RANGO'
                
                # Estado
                if fecha_estimada and fecha_estimada < hoy:
                    estado = '❌ VENCIDO'
                else:
                    estado = '✅ A TIEMPO'
                
                # Agregar a resultados
                resultados.append({
                    'Unidad': unidad,
                    'KM_Actual': km_actual,
                    'Ultimo_Preventivo_KM': ultimo_km_preventivo,
                    'Fecha_Ultimo_Preventivo': ultima_fecha_preventivo,
                    'Intervalo': intervalo,
                    'KM_Esperado': km_esperado,
                    'KM_Recorridos': km_recorridos,
                    'Porc_Avance': porc_avance,
                    'Alerta': alerta,
                    'KM_Promedio_Diario': km_promedio_diario,
                    'Dias_Faltantes': dias_faltantes,
                    'Fecha_Estimada': fecha_estimada,
                    'Estado': estado,
                    'Fecha_KM_Actual': fecha_km_actual
                })
        
        # Convertir a DataFrame
        if resultados:
            df_resultados = pd.DataFrame(resultados)
            
            # Formatear para presentación
            df_final = pd.DataFrame()
            df_final['Unidad'] = df_resultados['Unidad']
            df_final['KM Actual'] = df_resultados['KM_Actual'].apply(lambda x: f"{x:,.1f}" if pd.notna(x) else "")
            df_final['Fecha Último Kilometraje'] = df_resultados['Fecha_KM_Actual'].apply(
                lambda x: x.strftime('%d/%m/%Y %H:%M') if pd.notna(x) else ""
            )
            df_final['Último Preventivo (km)'] = df_resultados['Ultimo_Preventivo_KM'].apply(
                lambda x: f"{x:,.0f}" if pd.notna(x) else ""
            )
            df_final['Fecha Último Preventivo'] = df_resultados['Fecha_Ultimo_Preventivo'].apply(
                lambda x: x.strftime('%d/%m/%Y %H:%M') if pd.notna(x) else ""
            )
            df_final['Intervalo Preventivo'] = df_resultados['Intervalo'].apply(
                lambda x: f"{x:,.0f}" if pd.notna(x) else ""
            )
            df_final['KM Esperado Mantenimiento'] = df_resultados['KM_Esperado'].apply(
                lambda x: f"{x:,.0f}" if pd.notna(x) else ""
            )
            df_final['KM Recorridos'] = df_resultados['KM_Recorridos'].apply(
                lambda x: f"{x:,.0f}" if pd.notna(x) else ""
            )
            df_final['% Avance'] = df_resultados['Porc_Avance'].apply(
                lambda x: f"{x:.2f}%" if pd.notna(x) else "0%"
            )
            df_final['Alerta'] = df_resultados['Alerta']
            df_final['KM/Día Promedio'] = df_resultados['KM_Promedio_Diario'].apply(
                lambda x: f"{x:.1f}" if pd.notna(x) else ""
            )
            df_final['Días Faltantes'] = df_resultados['Dias_Faltantes'].apply(
                lambda x: f"{int(x)}" if pd.notna(x) and not np.isnan(x) else ""
            )
            
            def formatear_fecha_segura(fecha):
                if fecha is None or pd.isna(fecha):
                    return ""
                try:
                    if isinstance(fecha, (datetime, pd.Timestamp)):
                        return fecha.strftime('%d/%m/%Y %H:%M')
                    else:
                        return str(fecha)
                except:
                    return ""
            
            df_final['Fecha Estimada Mantenimiento'] = df_resultados['Fecha_Estimada'].apply(formatear_fecha_segura)
            df_final['Estado'] = df_resultados['Estado']
            
            # Ordenar por alerta y días faltantes
            orden_alerta = {'🔴 ATRASADO': 0, '🟡 PROXIMO': 1, '🟢 EN RANGO': 2}
            df_final['Orden'] = df_final['Alerta'].map(orden_alerta)
            
            # Convertir Días Faltantes a numérico para ordenar
            def convertir_dias_faltantes(valor):
                try:
                    return float(str(valor).replace(',', ''))
                except:
                    return np.nan
            
            df_final['Dias_Faltantes_Num'] = df_final['Días Faltantes'].apply(convertir_dias_faltantes)
            df_final = df_final.sort_values(['Orden', 'Dias_Faltantes_Num'])
            df_final = df_final.drop(columns=['Orden', 'Dias_Faltantes_Num'])
            
            # DataFrame completo para cálculos internos
            df_completo = df_resultados.copy()
            
            return df_final, df_completo, df_tipo_unidad
        else:
            return pd.DataFrame(), pd.DataFrame(), df_tipo_unidad
            
    except Exception as e:
        st.error(f"❌ Error en cálculo mejorado: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# ============================================
# 5. INTERFAZ PRINCIPAL CON MEJORAS
# ============================================

st.set_page_config(
    page_title="Dashboard Mantenimientos",
    page_icon="🚗",
    layout="wide"
)

st_autorefresh(interval=15 * 60 * 1000, key="datarefresh")

st.title("🚗 DASHBOARD DE MANTENIMIENTOS")
st.markdown(f"**Actualizado:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# Cargar datos mejorados
with st.spinner("🔄 Cargando datos mejorados..."):
    df_final, df_completo, df_tipo_unidad = cargar_y_calcular_mejorado()

if df_final.empty:
    st.error("⚠️ No se pudieron cargar los datos.")
else:
    # Crear vistas de calendario
    df_calendario = crear_vista_calendario(df_final, df_tipo_unidad)
    df_por_mes = crear_vista_por_mes(df_calendario, meses_a_mostrar=6)
    df_por_semana = crear_vista_semanal(df_calendario, semanas_a_mostrar=8)
    
    # Resumen general
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Unidades", len(df_final))
    with col2:
        atrasados = len(df_final[df_final['Alerta'] == '🔴 ATRASADO'])
        st.metric("Atrasados", atrasados, delta=f"-{atrasados}" if atrasados > 0 else "0")
    with col3:
        proximos = len(df_final[df_final['Alerta'] == '🟡 PROXIMO'])
        st.metric("Próximos", proximos)
    with col4:
        en_rango = len(df_final[df_final['Alerta'] == '🟢 EN RANGO'])
        st.metric("En Rango", en_rango)
    
    # Pestañas principales
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Tabla Principal", 
        "📅 Vista por Mes", 
        "🗓️ Vista por Semana", 
        "📊 Calendario Detallado"
    ])
    
    with tab1:
        st.subheader("📋 TABLA PRINCIPAL DE MANTENIMIENTOS")
        
        # Filtros
        st.sidebar.header("🔍 FILTROS - TABLA PRINCIPAL")
        alertas = sorted(df_final['Alerta'].unique())
        estados = sorted([e for e in df_final['Estado'].unique() if e != ''])
        unidades = sorted(df_final['Unidad'].unique())
        
        alertas_sel = st.sidebar.multiselect("Alerta:", alertas, default=alertas, key="filtro_alerta")
        estados_sel = st.sidebar.multiselect("Estado:", estados, default=estados, key="filtro_estado")
        unidades_sel = st.sidebar.multiselect("Unidad:", unidades, default=unidades, key="filtro_unidad")
        
        # Aplicar filtros
        df_filtrado = df_final[
            (df_final['Alerta'].isin(alertas_sel)) &
            (df_final['Estado'].isin(estados_sel)) &
            (df_final['Unidad'].isin(unidades_sel))
        ]
        
        st.sidebar.metric("Unidades filtradas", len(df_filtrado))
        
        # Mostrar tabla
        columnas_mostrar = [
            'Unidad', 'KM Actual', 'Fecha Último Kilometraje',
            'Último Preventivo (km)', 'Fecha Último Preventivo', 
            'Intervalo Preventivo', 'KM Esperado Mantenimiento', 
            'KM Recorridos', '% Avance', 'Alerta', 'KM/Día Promedio', 
            'Días Faltantes', 'Fecha Estimada Mantenimiento', 'Estado'
        ]
        
        st.dataframe(
            df_filtrado[columnas_mostrar],
            use_container_width=True,
            hide_index=True
        )
        
        # Exportar
        if len(df_filtrado) > 0:
            csv_data = df_filtrado.to_csv(index=False, sep=';', encoding='utf-8-sig')
            st.download_button(
                label="📥 Descargar Tabla Principal",
                data=csv_data,
                file_name=f"tabla_principal_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="descarga_principal"
            )
    
    with tab2:
        # VISTA POR MES (CON TIPO UNIDAD)
        st.subheader("📅 VISTA POR MES - PRÓXIMOS MANTENIMIENTOS")
        
        if not df_por_mes.empty:
            # Mostrar resumen por mes
            for _, mes_data in df_por_mes.iterrows():
                with st.expander(f"**{mes_data['Mes_Nombre']} {mes_data['Año']} - {mes_data['Cantidad_Unidades']} unidades**", expanded=True):
                    # Filtrar unidades de este mes
                    unidades_mes = df_calendario[
                        (df_calendario['Mes'] == mes_data['Mes']) & 
                        (df_calendario['Año'] == mes_data['Año'])
                    ].sort_values('Fecha_Estimada_DT')
                    
                    # Mostrar por semana dentro del mes
                    semanas_mes = sorted(unidades_mes['Semana'].unique())
                    
                    for semana in semanas_mes:
                        unidades_semana = unidades_mes[unidades_mes['Semana'] == semana]
                        
                        # Encontrar fechas de la semana
                        primera_fecha = unidades_semana['Fecha_Estimada_DT'].min().strftime('%d/%m')
                        ultima_fecha = unidades_semana['Fecha_Estimada_DT'].max().strftime('%d/%m')
                        
                        st.write(f"**Semana {semana} ({primera_fecha} - {ultima_fecha}):** {len(unidades_semana)} unidades")
                        
                        # Mostrar tabla de unidades CON TIPO UNIDAD
                        columnas_calendario = ['Unidad']
                        if 'Tipo Unidad' in unidades_semana.columns:
                            columnas_calendario.append('Tipo Unidad')
                        columnas_calendario.extend(['Fecha Estimada Mantenimiento', 'Alerta', 'Estado', 'KM/Día Promedio'])
                        
                        st.dataframe(
                            unidades_semana[columnas_calendario],
                            use_container_width=True,
                            hide_index=True
                        )
        else:
            st.info("No hay mantenimientos programados para los próximos meses")
        
        # Gráfico de barras por mes
        if not df_por_mes.empty:
            st.subheader("📊 GRÁFICO POR MES")
            
            fig_mes = px.bar(
                df_por_mes,
                x='Mes_Nombre',
                y='Cantidad_Unidades',
                color='Cantidad_Unidades',
                text='Cantidad_Unidades',
                title='Mantenimientos Programados por Mes',
                color_continuous_scale='viridis'
            )
            fig_mes.update_traces(textposition='outside')
            st.plotly_chart(fig_mes, use_container_width=True)
    
    with tab3:
        # VISTA POR SEMANA (CON TIPO UNIDAD)
        st.subheader("🗓️ VISTA POR SEMANA - PRÓXIMAS 8 SEMANAS")
        
        if not df_por_semana.empty:
            # Mostrar por semana
            for _, semana_data in df_por_semana.iterrows():
                with st.expander(f"**Semana {semana_data['Semana']} ({semana_data['Rango_Semana']}) - {semana_data['Cantidad_Unidades']} unidades**", expanded=True):
                    # Filtrar unidades de esta semana
                    unidades_semana = df_calendario[
                        (df_calendario['Semana'] == semana_data['Semana']) & 
                        (df_calendario['Año'] == semana_data['Año'])
                    ].sort_values('Fecha_Estimada_DT')
                    
                    # Mostrar por día
                    unidades_semana['Dia_Semana'] = unidades_semana['Fecha_Estimada_DT'].dt.day_name()
                    dias_semana = unidades_semana.groupby(['Dia', 'Dia_Semana']).agg({
                        'Unidad': 'count',
                        'Alerta': lambda x: list(x)
                    }).reset_index()
                    
                    # Tabla por día
                    for _, dia_data in dias_semana.iterrows():
                        st.write(f"**{dia_data['Dia_Semana']} {dia_data['Dia']}:** {dia_data['Unidad']} unidades")
                        
                        # Unidades de este día
                        unidades_dia = unidades_semana[unidades_semana['Dia'] == dia_data['Dia']]
                        
                        # Mostrar tabla CON TIPO UNIDAD
                        columnas_calendario = ['Unidad']
                        if 'Tipo Unidad' in unidades_dia.columns:
                            columnas_calendario.append('Tipo Unidad')
                        columnas_calendario.extend(['Fecha Estimada Mantenimiento', 'Alerta', 'Estado'])
                        
                        st.dataframe(
                            unidades_dia[columnas_calendario],
                            use_container_width=True,
                            hide_index=True
                        )
        else:
            st.info("No hay mantenimientos programados para las próximas semanas")
        
        # Gráfico por semana
        if not df_por_semana.empty:
            st.subheader("📈 GRÁFICO POR SEMANA")
            
            fig_semana = px.bar(
                df_por_semana,
                x='Rango_Semana',
                y='Cantidad_Unidades',
                color='Cantidad_Unidades',
                text='Cantidad_Unidades',
                title='Mantenimientos Programados por Semana',
                color_continuous_scale='plasma'
            )
            fig_semana.update_traces(textposition='outside')
            st.plotly_chart(fig_semana, use_container_width=True)
    
    with tab4:
        # CALENDARIO DETALLADO (CON TIPO UNIDAD)
        st.subheader("📊 CALENDARIO DETALLADO - VISTA COMPLETA")
        
        if not df_calendario.empty:
            # Selector de mes
            meses_disponibles = sorted(df_calendario['Mes_Año'].unique())
            mes_seleccionado = st.selectbox(
                "Seleccionar mes para ver detalle:",
                meses_disponibles,
                index=0 if len(meses_disponibles) > 0 else 0,
                key="selector_mes"
            )
            
            if mes_seleccionado:
                # Filtrar mes seleccionado
                df_mes_seleccionado = df_calendario[df_calendario['Mes_Año'] == mes_seleccionado].copy()
                
                # Obtener nombre del mes
                fecha_ejemplo = df_mes_seleccionado['Fecha_Estimada_DT'].iloc[0]
                nombre_mes = fecha_ejemplo.strftime('%B %Y')
                
                st.write(f"### 📅 {nombre_mes}")
                
                # Crear columna combinada Unidad + Tipo Unidad
                if 'Tipo Unidad' in df_mes_seleccionado.columns:
                    df_mes_seleccionado['Unidad_Completa'] = df_mes_seleccionado.apply(
                        lambda row: f"{row['Unidad']} ({row['Tipo Unidad']})" if row['Tipo Unidad'] and row['Tipo Unidad'] != 'No especificado' else row['Unidad'],
                        axis=1
                    )
                else:
                    df_mes_seleccionado['Unidad_Completa'] = df_mes_seleccionado['Unidad']
                
                # Agrupar por día
                df_por_dia = df_mes_seleccionado.groupby(['Dia', 'Fecha_Estimada_DT']).agg({
                    'Unidad_Completa': lambda x: ', '.join(sorted(x)),
                    'Alerta': lambda x: list(x)
                }).reset_index()
                
                # Agregar columna de cantidad
                df_por_dia['Unidades'] = df_por_dia['Unidad_Completa'].apply(lambda x: len(x.split(', ')) if x else 0)
                
                # Ordenar por día
                df_por_dia = df_por_dia.sort_values('Dia')
                
                # Mostrar como tabla de calendario CON TIPO UNIDAD
                st.dataframe(
                    df_por_dia[['Dia', 'Unidades', 'Unidad_Completa']],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Dia": st.column_config.NumberColumn(format="%d"),
                        "Unidades": st.column_config.NumberColumn(format="%d"),
                        "Unidad_Completa": st.column_config.TextColumn(
                            "Unidad (Tipo)", 
                            width="large"
                        )
                    }
                )
                
                # Gráfico de calendario (heatmap)
                st.subheader("🎯 VISUALIZACIÓN DE CALENDARIO")
                
                # Crear matriz para heatmap
                dias_mes = range(1, 32)
                data_heatmap = []
                
                for dia in dias_mes:
                    unidades_dia = df_mes_seleccionado[df_mes_seleccionado['Dia'] == dia]
                    count = len(unidades_dia)
                    
                    # Determinar color por alertas
                    if count > 0:
                        alertas_dia = unidades_dia['Alerta'].tolist()
                        if '🔴 ATRASADO' in alertas_dia:
                            color = 'red'
                        elif '🟡 PROXIMO' in alertas_dia:
                            color = 'orange'
                        else:
                            color = 'green'
                    else:
                        color = 'lightgray'
                    
                    data_heatmap.append({
                        'Dia': dia,
                        'Unidades': count,
                        'Color': color
                    })
                
                df_heatmap = pd.DataFrame(data_heatmap)
                
                # Crear heatmap
                fig_calendario = go.Figure(data=go.Heatmap(
                    z=df_heatmap['Unidades'].values.reshape(1, -1),
                    x=df_heatmap['Dia'].values,
                    colorscale='Reds',
                    showscale=True,
                    hoverinfo='x+z',
                    text=df_heatmap['Unidades'].values.reshape(1, -1),
                    texttemplate='%{text}',
                    textfont={"size": 16}
                ))
                
                fig_calendario.update_layout(
                    title=f'Calendario de Mantenimientos - {nombre_mes}',
                    xaxis_title='Día del Mes',
                    yaxis=dict(showticklabels=False),
                    height=200
                )
                
                st.plotly_chart(fig_calendario, use_container_width=True)
                
                # Mostrar vista detallada por día
                st.subheader("📋 DETALLE POR DÍA")
                
                for dia in sorted(df_mes_seleccionado['Dia'].unique()):
                    unidades_dia = df_mes_seleccionado[df_mes_seleccionado['Dia'] == dia]
                    if len(unidades_dia) > 0:
                        with st.expander(f"Día {dia} - {len(unidades_dia)} unidades", expanded=False):
                            columnas_detalle = ['Unidad']
                            if 'Tipo Unidad' in unidades_dia.columns:
                                columnas_detalle.append('Tipo Unidad')
                            columnas_detalle.extend(['Fecha Estimada Mantenimiento', 'Alerta', 'Estado'])
                            
                            st.dataframe(
                                unidades_dia[columnas_detalle],
                                use_container_width=True,
                                hide_index=True
                            )
                
                # Exportar datos del mes
                csv_mes = df_mes_seleccionado.to_csv(index=False, sep=';', encoding='utf-8-sig')
                st.download_button(
                    label=f"📥 Descargar Datos de {nombre_mes}",
                    data=csv_mes,
                    file_name=f"mantenimientos_{mes_seleccionado}.csv",
                    mime="text/csv",
                    key=f"descarga_{mes_seleccionado}"
                )
        else:
            st.info("No hay datos de calendario disponibles")

# ============================================
# 6. PIE DE PÁGINA
# ============================================

st.markdown("---")
st.caption(f"""
🚗 **Dashboard de Mantenimientos ** | 
{len(df_final) if not df_final.empty else 0} unidades monitoreadas | 
{len(df_calendario) if not df_calendario.empty else 0} con fecha estimada | 
Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
""")
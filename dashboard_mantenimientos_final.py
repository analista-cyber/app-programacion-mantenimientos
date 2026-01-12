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
# 1. URLs (REEMPLAZA)
# ============================================

URL_MANTENIMIENTOS = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSI2Ujvaj6GKzQDfhFXZbZ3bzRRtUNluxlPLVQuyruijv4ILq6jMWasYR44BRr4lLxlUg9ZBU28FUek/pub?gid=424956520&single=true&output=csv"
URL_HISTORICO = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTIRp3QEfyTn35RIDPHN0sS9mQYCNeQCrJdhZY_DGLkkUfcwaXPCvIlkQvWJ3OzDrKNLmUZQ9HeL83h/pub?gid=216802721&single=true&output=csv"
URL_CATALOGO = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSI2Ujvaj6GKzQDfhFXZbZ3bzRRtUNluxlPLVQuyruijv4ILq6jMWasYR44BRr4lLxlUg9ZBU28FUek/pub?gid=41327256&single=true&output=csv"

# ============================================
# 2. FUNCIONES PARA CALENDARIO
# ============================================

def crear_vista_calendario(df_final):
    """Crea vista de calendario por mes"""
    
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
    
    return df_con_fecha

def crear_vista_por_mes(df_con_fecha, meses_a_mostrar=6):
    """Crea vista agrupada por mes"""
    
    if len(df_con_fecha) == 0:
        return pd.DataFrame()
    
    hoy = datetime.now()
    meses = []
    
    # Generar próximos N meses
    for i in range(meses_a_mostrar):
        fecha = hoy + relativedelta(months=i)
        meses.append(fecha.strftime('%Y-%m'))
    
    # Filtrar solo los próximos meses
    df_futuro = df_con_fecha[df_con_fecha['Mes_Año'].isin(meses)].copy()
    
    if len(df_futuro) == 0:
        return pd.DataFrame()
    
    # Agrupar por mes
    vista_mes = df_futuro.groupby(['Año', 'Mes', 'Mes_Nombre', 'Mes_Año']).agg({
        'Unidad': 'count',
        'Alerta': lambda x: list(x),
        'Estado': lambda x: list(x)
    }).reset_index()
    
    vista_mes = vista_mes.rename(columns={'Unidad': 'Cantidad_Unidades'})
    
    # Ordenar por fecha
    vista_mes = vista_mes.sort_values(['Año', 'Mes'])
    
    return vista_mes

def crear_vista_semanal(df_con_fecha, semanas_a_mostrar=8):
    """Crea vista por semana"""
    
    if len(df_con_fecha) == 0:
        return pd.DataFrame()
    
    hoy = datetime.now()
    semana_actual = hoy.isocalendar().week
    año_actual = hoy.year
    
    # Filtrar próximas semanas
    df_futuro = df_con_fecha[
        (df_con_fecha['Año'] == año_actual) & 
        (df_con_fecha['Semana'] >= semana_actual)
    ].copy()
    
    if len(df_futuro) == 0:
        return pd.DataFrame()
    
    # Limitar a N semanas
    semanas = sorted(df_futuro['Semana'].unique())[:semanas_a_mostrar]
    df_futuro = df_futuro[df_futuro['Semana'].isin(semanas)]
    
    # Agrupar por semana
    vista_semana = df_futuro.groupby(['Año', 'Semana']).agg({
        'Unidad': 'count',
        'Alerta': lambda x: list(x),
        'Fecha_Estimada_DT': 'min'
    }).reset_index()
    
    vista_semana = vista_semana.rename(columns={'Unidad': 'Cantidad_Unidades'})
    
    # Agregar rango de fechas de la semana
    def obtener_rango_semana(fila):
        year = fila['Año']
        week = fila['Semana']
        
        # Encontrar primer día de la semana (lunes)
        first_day = datetime.fromisocalendar(year, week, 1)
        last_day = first_day + timedelta(days=6)
        
        return f"{first_day.strftime('%d/%m')} - {last_day.strftime('%d/%m')}"
    
    vista_semana['Rango_Semana'] = vista_semana.apply(obtener_rango_semana, axis=1)
    vista_semana = vista_semana.sort_values(['Año', 'Semana'])
    
    return vista_semana

# ============================================
# 3. FUNCIÓN PRINCIPAL (LA MISMA QUE FUNCIONA)
# ============================================

def calcular_km_promedio_diario_corregido(df_hist, unidad):
    """Calcula KM promedio diario agrupando por día"""
    df_unidad = df_hist[df_hist['Nombre Unidad'] == unidad].copy()
    
    if len(df_unidad) < 2:
        return 100
    
    # Agrupar por día
    df_unidad['Fecha_Dia'] = df_unidad['Fecha'].dt.date
    df_por_dia = df_unidad.sort_values('Fecha', ascending=False)
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

@st.cache_data(ttl=300)
def cargar_y_calcular():
    """Función principal (la que ya funciona)"""
    
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
        
        # Filtrar preventivos
        if 'TIPO DE SERVICIO' in df_mto.columns:
            mask = df_mto['TIPO DE SERVICIO'].str.contains('Preventivo', case=False, na=False)
            df_mto_preventivo = df_mto[mask].copy()
        else:
            df_mto_preventivo = df_mto.copy()
        
        df_mto_preventivo = df_mto_preventivo.sort_values('FECHA', ascending=False)
        df_ultimo_mto = df_mto_preventivo.groupby('Nombre Unidad').first().reset_index()
        df_ultimo_mto = df_ultimo_mto.rename(columns={'Nombre Unidad': 'Nombre_Unidad'})
        
        # KM actual
        df_hist_reciente = df_hist.sort_values('Fecha', ascending=False)
        df_km_actual = df_hist_reciente.groupby('Nombre Unidad').first().reset_index()
        df_km_actual = df_km_actual.rename(columns={
            'Nombre Unidad': 'Nombre_Unidad',
            'Kilometraje KM': 'KM_Actual',
            'Fecha': 'Fecha_KM_Actual'
        })
        
        # Combinar
        df = pd.merge(
            df_ultimo_mto[['Nombre_Unidad', 'FECHA', 'KILOMETRAJE']],
            df_km_actual[['Nombre_Unidad', 'KM_Actual', 'Fecha_KM_Actual']],
            on='Nombre_Unidad',
            how='left'
        )
        
        df = pd.merge(
            df,
            df_cat.rename(columns={'Nombre Unidad': 'Nombre_Unidad'})[['Nombre_Unidad', 'INTERVALO']],
            on='Nombre_Unidad',
            how='left'
        )
        
        hoy = datetime.now()
        
        # Calcular KM promedio
        unidades = df['Nombre_Unidad'].unique()
        km_promedio_dict = {}
        
        for unidad in unidades:
            promedio = calcular_km_promedio_diario_corregido(df_hist, unidad)
            km_promedio_dict[unidad] = promedio
        
        df['KM_Promedio_Diario'] = df['Nombre_Unidad'].map(km_promedio_dict)
        
        # Cálculos básicos
        df['Dias_Transcurridos'] = (hoy - df['FECHA']).dt.days
        df['KM_Recorridos'] = df['KM_Actual'] - df['KILOMETRAJE']
        df['KM_Esperado'] = df['KILOMETRAJE'] + df['INTERVALO']
        
        # % Avance
        df['%_Avance'] = np.where(
            df['INTERVALO'] > 0,
            (df['KM_Recorridos'] / df['INTERVALO']) * 100,
            0
        )
        
        # Días Faltantes
        def calcular_dias_faltantes_seguro(fila):
            try:
                km_rec = fila['KM_Recorridos']
                intervalo = fila['INTERVALO']
                prom_km_dia = fila['KM_Promedio_Diario']
                
                if pd.isna(km_rec) or pd.isna(intervalo) or pd.isna(prom_km_dia):
                    return np.nan
                
                if prom_km_dia <= 0:
                    return np.nan
                
                km_faltantes = intervalo - km_rec
                dias_faltantes = km_faltantes / prom_km_dia
                
                return round(dias_faltantes, 0)
            except:
                return np.nan
        
        df['Dias_Faltantes'] = df.apply(calcular_dias_faltantes_seguro, axis=1)
        
        # Fecha Estimada
        def calcular_fecha_estimada_segura(fila):
            try:
                dias_faltantes = fila['Dias_Faltantes']
                
                if pd.isna(dias_faltantes):
                    return None
                
                dias_faltantes = float(dias_faltantes)
                if np.isnan(dias_faltantes) or np.isinf(dias_faltantes):
                    return None
                
                return hoy + timedelta(days=int(dias_faltantes))
            except:
                return None
        
        df['Fecha_Estimada'] = df.apply(calcular_fecha_estimada_segura, axis=1)
        
        # Alertas
        condiciones_alerta = [
            df['%_Avance'] >= 100,
            (df['%_Avance'] >= 80) & (df['%_Avance'] < 100),
            df['%_Avance'] < 80
        ]
        
        df['Alerta'] = np.select(
            condiciones_alerta,
            ['🔴 ATRASADO', '🟡 PROXIMO', '🟢 EN RANGO'],
            default='🟢 EN RANGO'
        )
        
        # Estado
        def calcular_estado_seguro(fila):
            try:
                fecha_est = fila['Fecha_Estimada']
                
                if fecha_est is None or pd.isna(fecha_est):
                    return ''
                
                if not isinstance(fecha_est, (datetime, pd.Timestamp)):
                    return ''
                
                if fecha_est < hoy:
                    return '❌ VENCIDO'
                else:
                    return '✅ A TIEMPO'
            except:
                return ''
        
        df['Estado'] = df.apply(calcular_estado_seguro, axis=1)
        
        # Formatear
        df['KM_Promedio_Diario'] = df['KM_Promedio_Diario'].round(1)
        df['%_Avance'] = df['%_Avance'].round(2)
        df['Dias_Faltantes'] = pd.to_numeric(df['Dias_Faltantes'], errors='coerce').round(0)
        
        # DataFrame final
        df_final = pd.DataFrame()
        
        df_final['Unidad'] = df['Nombre_Unidad']
        df_final['KM Actual'] = df['KM_Actual'].apply(lambda x: f"{x:,.1f}" if pd.notna(x) else "")
        df_final['Último Preventivo (km)'] = df['KILOMETRAJE'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
        df_final['Fecha Último Preventivo'] = df['FECHA'].apply(
            lambda x: x.strftime('%d/%m/%Y %H:%M') if pd.notna(x) else ""
        )
        df_final['Intervalo Preventivo'] = df['INTERVALO'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
        df_final['KM Esperado Mantenimiento'] = df['KM_Esperado'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
        df_final['KM Recorridos'] = df['KM_Recorridos'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
        df_final['% Avance'] = df['%_Avance'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "0%")
        df_final['Alerta'] = df['Alerta']
        df_final['KM/Día Promedio'] = df['KM_Promedio_Diario'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")
        df_final['Días Faltantes'] = df['Dias_Faltantes'].apply(lambda x: f"{int(x)}" if pd.notna(x) else "")
        
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
        
        df_final['Fecha Estimada Mantenimiento'] = df['Fecha_Estimada'].apply(formatear_fecha_segura)
        df_final['Estado'] = df['Estado']
        
        # Ordenar
        orden_alerta = {'🔴 ATRASADO': 0, '🟡 PROXIMO': 1, '🟢 EN RANGO': 2}
        df_final['Orden'] = df_final['Alerta'].map(orden_alerta)
        df_final['Dias_Faltantes_Num'] = pd.to_numeric(df_final['Días Faltantes'], errors='coerce')
        df_final = df_final.sort_values(['Orden', 'Dias_Faltantes_Num'])
        df_final = df_final.drop(columns=['Orden', 'Dias_Faltantes_Num'])
        
        return df_final, df
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# ============================================
# 4. INTERFAZ PRINCIPAL CON PESTAÑAS
# ============================================

st.set_page_config(
    page_title="Dashboard Mantenimientos",
    page_icon="🚗",
    layout="wide"
)

st_autorefresh(interval=15 * 60 * 1000, key="datarefresh")

# Título principal
st.title("🚗 DASHBOARD DE MANTENIMIENTOS ")
st.markdown(f"Actualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# Cargar datos
with st.spinner("🔄 Cargando datos..."):
    df_final, df_completo = cargar_y_calcular()

if df_final.empty:
    st.error("⚠️ No se pudieron cargar los datos.")
else:
    # ============================================
    # 5. CREAR VISTAS DE CALENDARIO
    # ============================================
    
    df_calendario = crear_vista_calendario(df_final)
    df_por_mes = crear_vista_por_mes(df_calendario, meses_a_mostrar=6)
    df_por_semana = crear_vista_semanal(df_calendario, semanas_a_mostrar=8)
    
    # ============================================
    # 6. PESTAÑAS PRINCIPALES
    # ============================================
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Tabla Principal", 
        "📅 Vista por Mes", 
        "🗓️ Vista por Semana", 
        "📊 Calendario Detallado"
    ])
    
    with tab1:
        # TABLA PRINCIPAL (LA QUE YA TENÍAS)
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
            'Unidad', 'KM Actual', 'Último Preventivo (km)', 
            'Fecha Último Preventivo', 'Intervalo Preventivo',
            'KM Esperado Mantenimiento', 'KM Recorridos', '% Avance',
            'Alerta', 'KM/Día Promedio', 'Días Faltantes', 
            'Fecha Estimada Mantenimiento', 'Estado'
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
        # VISTA POR MES
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
                        
                        # Mostrar tabla de unidades
                        st.dataframe(
                            unidades_semana[['Unidad', 'Fecha Estimada Mantenimiento', 'Alerta', 'Estado', 'KM/Día Promedio']],
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
        # VISTA POR SEMANA
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
                        st.dataframe(
                            unidades_dia[['Unidad', 'Fecha Estimada Mantenimiento', 'Alerta', 'Estado']],
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
        # CALENDARIO DETALLADO
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
                
                # CORRECCIÓN: Agrupar correctamente sin la columna 'Cantidad'
                df_por_dia = df_mes_seleccionado.groupby(['Dia', 'Fecha_Estimada_DT']).agg({
                    'Unidad': lambda x: ', '.join(sorted(x)),
                    'Alerta': lambda x: list(x)
                }).reset_index()
                
                # Agregar columna de cantidad
                df_por_dia['Unidades'] = df_por_dia['Unidad'].apply(lambda x: len(x.split(', ')) if x else 0)
                
                # Ordenar por día
                df_por_dia = df_por_dia.sort_values('Dia')
                
                # Mostrar como tabla de calendario
                st.dataframe(
                    df_por_dia[['Dia', 'Unidades', 'Unidad']],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Dia": st.column_config.NumberColumn(format="%d"),
                        "Unidades": st.column_config.NumberColumn(format="%d"),
                        "Unidad": st.column_config.TextColumn(width="large")
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
# 8. PIE DE PÁGINA
# ============================================

st.markdown("---")
st.caption(f"""
🚗 **Dashboard de Mantenimientos** | 
{len(df_final) if not df_final.empty else 0} unidades monitoreadas | 
{len(df_calendario) if not df_calendario.empty else 0} con fecha estimada | 
Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
""")